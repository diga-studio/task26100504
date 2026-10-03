"""MindLoom 协同文本内核测试。

运行：python3 -m pytest tests/ -v
"""

import itertools
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindloom import Replica  # noqa: E402


def wire(message):
    """模拟网络：深拷贝并验证消息可 JSON 序列化。"""
    return json.loads(json.dumps(message))


def broadcast(replicas, message):
    for r in replicas:
        r.receive(wire(message))


# ----------------------------------------------------------------------
# 1. 反向输入例子：穷举 4 条编辑消息的全部投递排列，并插入重复投递
# ----------------------------------------------------------------------

def test_reverse_input_all_permutations_with_duplicates():
    results = set()
    for perm in itertools.permutations(range(4)):
        for dup in range(4):
            a, b = Replica("甲"), Replica("乙")
            for m in (a.insert(0, "L"), a.insert(1, "R")):
                b.receive(wire(m))
            edits = [a.insert(1, "x"), a.insert(1, "y"),
                     b.insert(1, "u"), b.insert(1, "v")]
            order = [edits[i] for i in perm]
            order.insert(dup, wire(edits[dup]))  # 重复投递一条
            for m in order:
                broadcast([a, b], m)
            assert a.text() == b.text()
            results.add(a.text())
    # 只能是两种之一，且与投递顺序、重复次数无关
    assert results == {"LvuyxR"} or results == {"LyxvuR"}


def runs_contiguous(text, fragment):
    """fragment 的字符在 text 中构成连续子串且保持顺序。"""
    return fragment in text


# ----------------------------------------------------------------------
# 2. 两端混合插入：片段不交错
# ----------------------------------------------------------------------

def _mixed_end_scripts():
    """返回 (甲的插入序列, 乙的插入序列, 甲片段, 乙片段)。

    双方从 "LR" 的同一缝隙出发，在各自新增文本内部及两端继续插入。
    """
    # 甲：右插 a -> 右插 b -> 左插 c -> 中间插 d  => 片段 "cadb"
    ops_a = [(1, "a"), (2, "b"), (1, "c"), (3, "d")]
    frag_a = "cadb"
    # 乙：左插 x -> 左插 y -> 右插 z -> 中间插 w  => 片段 "ywxz"
    ops_b = [(1, "x"), (1, "y"), (3, "z"), (2, "w")]
    frag_b = "ywxz"
    return ops_a, ops_b, frag_a, frag_b


def test_mixed_end_inserts_no_interleaving():
    ops_a, ops_b, frag_a, frag_b = _mixed_end_scripts()
    results = set()
    for perm in itertools.permutations(range(8)):
        a, b = Replica("甲"), Replica("乙")
        for m in (a.insert(0, "L"), a.insert(1, "R")):
            b.receive(wire(m))
        msgs = []
        for pos, t in ops_a:
            msgs.append(a.insert(pos, t))
        for pos, t in ops_b:
            msgs.append(b.insert(pos, t))
        for i in perm:
            broadcast([a, b], msgs[i])
        broadcast([a, b], msgs[perm[0]])  # 重复投递一条
        assert a.text() == b.text()
        results.add(a.text())
    assert len(results) == 1
    merged = results.pop()
    assert merged.startswith("L") and merged.endswith("R")
    body = merged[1:-1]
    assert runs_contiguous(body, frag_a), (body, frag_a)
    assert runs_contiguous(body, frag_b), (body, frag_b)
    assert len(body) == len(frag_a) + len(frag_b)


def test_pure_right_append_no_interleaving():
    results = set()
    for perm in itertools.permutations(range(4)):
        a, b = Replica("A"), Replica("B")
        for m in (a.insert(0, "L"), a.insert(1, "R")):
            b.receive(wire(m))
        edits = [a.insert(1, "x"), a.insert(2, "y"),
                 b.insert(1, "u"), b.insert(2, "v")]
        for i in perm:
            broadcast([a, b], edits[i])
        assert a.text() == b.text()
        results.add(a.text())
    assert len(results) == 1
    body = results.pop()[1:-1]
    assert ("xy" in body) and ("uv" in body)


# ----------------------------------------------------------------------
# 3. 删除与因果乱序
# ----------------------------------------------------------------------

def test_delete_arrives_before_create():
    a, b = Replica("A"), Replica("B")
    m1 = a.insert(0, "a")
    m2 = a.insert(1, "b")
    m3 = a.insert(2, "c")
    md = a.delete(1, 2)  # 删除 "b"
    b.receive(wire(md))  # 删除先于创建到达
    assert b.text() == ""
    b.receive(wire(m2))  # "b" 的创建到达：应立即是删除态
    assert b.text() == ""
    b.receive(wire(m1))
    b.receive(wire(m3))
    assert b.text() == "ac"
    b.receive(wire(md))  # 重复删除：幂等
    assert b.text() == "ac"


def test_insert_with_both_neighbors_deleted():
    a, b = Replica("A"), Replica("B")
    setup = [a.insert(0, "L"), a.insert(1, "R")]
    for m in setup:
        b.receive(wire(m))
    ins = b.insert(1, "m")       # m 的左右邻居是 L 和 R
    dele = a.delete(0, 2)        # 甲删除 L 和 R（未见 m）
    broadcast([a, b], dele)
    broadcast([a, b], ins)
    assert a.text() == b.text() == "m"  # 锚点被删，插入不丢失


def test_independent_updates_progress_while_dependency_missing():
    a, b = Replica("A"), Replica("B")
    m1 = a.insert(0, "a")
    m2 = a.insert(1, "b")
    m3 = a.insert(2, "c")  # 依赖 m2
    b.receive(wire(m3))    # 依赖缺失：缓冲，不阻塞
    assert b.text() == ""
    local = b.insert(0, "Z")  # 缺失依赖期间本地编辑照常生效
    assert b.text() == "Z"
    b.receive(wire(m2))       # 仍缺 m1
    assert b.text() == "Z"
    b.receive(wire(m1))       # 依赖补齐，缓冲重放
    assert b.text() == "Zabc"
    a.receive(wire(local))
    assert a.text() == "Zabc"


def test_delete_does_not_swallow_concurrent_insert():
    a, b = Replica("A"), Replica("B")
    for m in (a.insert(0, "x"), a.insert(1, "y"), a.insert(2, "z")):
        b.receive(wire(m))
    dele = a.delete(0, 3)      # 甲删除全部可见字符
    ins = b.insert(1, "Q")     # 乙并发地在中间插入
    broadcast([a, b], dele)
    broadcast([a, b], ins)
    assert a.text() == b.text() == "Q"


def test_same_value_chars_not_merged():
    a, b = Replica("A"), Replica("B")
    m1 = a.insert(0, "🐍")
    m2 = b.insert(0, "🐍")     # 同值并发插入：不得合并
    broadcast([a, b], m1)
    broadcast([a, b], m2)
    assert a.text() == b.text() == "🐍🐍"


# ----------------------------------------------------------------------
# 4. 固定随机种子三副本交错编辑：部分合流后继续编辑，最终收敛
# ----------------------------------------------------------------------

def test_three_replicas_random_convergence():
    rng = random.Random(20261005)
    replicas = [Replica("r1"), Replica("r2"), Replica("r3")]
    mirrors = {r.replica_id: "" for r in replicas}
    alphabet = "aabb中🐍𝕏"  # 含重复字符与非 BMP 字符
    delivered = {}  # id(msg) -> (消息, 已收到的副本 id 集合)

    def deliver(msg, targets):
        entry = delivered.setdefault(id(msg), (msg, set()))
        for r in targets:
            r.receive(wire(msg))
            entry[1].add(r.replica_id)

    for step in range(240):
        r = rng.choice(replicas)
        # 副本可能刚收到过远程消息，以其当前可见文本为基准校验切片语义
        s = r.text()
        mirrors[r.replica_id] = s
        if s and rng.random() < 0.3:
            start = rng.randrange(len(s))
            end = rng.randrange(start + 1, len(s) + 1)
            msg = r.delete(start, end)
            mirrors[r.replica_id] = s[:start] + s[end:]
        else:
            pos = rng.randrange(len(s) + 1)
            t = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 4)))
            msg = r.insert(pos, t)
            mirrors[r.replica_id] = s[:pos] + t + s[pos:]
        # 每次本地操作的即时结果必须等于字符串切片
        assert r.text() == mirrors[r.replica_id]
        delivered.setdefault(id(msg), (msg, set()))  # 登记，最终补齐
        # 随机向部分副本投递（含重复投递）；未收到的副本留待最终补齐
        if rng.random() < 0.7:
            targets = [x for x in replicas if rng.random() < 0.6]
            deliver(msg, targets)
            if rng.random() < 0.3:
                deliver(msg, targets)  # 重复投递

    # 部分合流后继续编辑（上面循环已交替进行），最后补齐全部消息
    for msg, got in delivered.values():
        deliver(msg, [r for r in replicas if r.replica_id not in got])
    all_msgs = [m for m, _ in delivered.values()]
    for msg in rng.sample(all_msgs, k=len(all_msgs)):  # 乱序重复一轮
        deliver(msg, replicas)

    texts = {r.text() for r in replicas}
    assert len(texts) == 1
