"""MindLoom 协同文本内核测试。运行：python3 -m pytest test_mindloom.py -v"""

import itertools
import random

from mindloom import TextReplica


def _synced_pair():
    """两个副本，A 插入初始文本 "LR" 并同步给 B。"""
    a, b = TextReplica("A"), TextReplica("B")
    b.receive(a.insert(0, "L"))
    b.receive(a.insert(1, "R"))
    assert a.text() == b.text() == "LR"
    return a, b


def _deliver(replica, message, times=1):
    for _ in range(times):
        replica.receive(message)


def test_reverse_input_all_permutations_with_duplicates():
    """题目示例：甲前插 x、y，乙前插 u、v，穷举 4! 种投递排列并重复投递。

    规则：片段按 replica_id 升序排列，A 的片段在前，故唯一结果为 LyxvuR。
    """
    results = set()
    for perm in itertools.permutations(range(4)):
        a, b = _synced_pair()
        msgs = [a.insert(1, "x"), a.insert(1, "y"),
                b.insert(1, "u"), b.insert(1, "v")]
        assert a.text() == "LyxR"
        assert b.text() == "LvuR"
        for i in perm:
            target = b if msgs[i]["replica"] == "A" else a
            _deliver(target, msgs[i], times=2)  # 重复投递必须幂等
        _deliver(a, msgs[2], times=1)           # 再补一轮乱序重复
        _deliver(b, msgs[0], times=1)
        assert a.text() == b.text(), f"perm={perm}: {a.text()!r} != {b.text()!r}"
        results.add(a.text())
    assert results == {"LyxvuR"}, results


def _assert_single_run(text, chars):
    """chars 中的字符必须在 text 中构成一个连续片段且顺序一致。"""
    idx = text.index(chars[0])
    assert text[idx:idx + len(chars)] == chars, (text, chars)


def test_mixed_append_prepend_no_interleaving():
    """两端混合插入：双方各自追加+前插，合并后片段不得交错。"""
    a, b = _synced_pair()
    a_msgs = [a.insert(1, "1"),              # L1R      （追加）
              a.insert(2, "2"),              # L12R     （追加）
              a.insert(1, "3"),              # L312R    （前插）
              a.insert(2, "4")]              # L3412R   （片段内部）
    b_msgs = [b.insert(1, "a"),              # LaR
              b.insert(2, "b"),              # LabR
              b.insert(1, "c")]              # LcabR
    assert a.text() == "L3412R"
    assert b.text() == "LcabR"

    rng = random.Random(7)
    outgoing = [(m, b) for m in a_msgs] + [(m, a) for m in b_msgs]
    rng.shuffle(outgoing)
    for msg, target in outgoing:
        _deliver(target, msg, times=rng.randint(1, 3))

    assert a.text() == b.text()
    final = a.text()
    assert final == "L3412cabR", final
    # 不只比较相等：显式检查两个片段各自连续且保持本地顺序
    _assert_single_run(final, "3412")
    _assert_single_run(final, "cab")


def test_pure_append_and_pure_prepend_runs():
    """纯追加与纯前插两种模式同样不交错。"""
    a, b = _synced_pair()
    a_msgs = [a.insert(1, "x"), a.insert(2, "y"), a.insert(3, "z")]   # LxyzR
    b_msgs = [b.insert(1, "p"), b.insert(1, "q"), b.insert(1, "r")]   # LrqpR
    for m in reversed(b_msgs):
        _deliver(a, m)
    for m in reversed(a_msgs):
        _deliver(b, m)
    assert a.text() == b.text() == "LxyzrqpR"


def test_delete_arrives_before_create():
    """删除消息先于被删字符的创建消息到达，最终仍一致。"""
    a, b = TextReplica("A"), TextReplica("B")
    m_ins = a.insert(0, "hi")
    m_del = a.delete(0, 1)            # 删除 'h'
    assert a.text() == "i"
    _deliver(b, m_del)                # 删除先到：目标未知，缓冲
    assert b.text() == ""
    _deliver(b, m_ins)                # 创建后到：'h' 落库即删
    assert b.text() == "i"
    _deliver(b, m_del, times=2)       # 重复删除幂等
    assert b.text() == "i"


def test_insert_between_two_deleted_neighbors():
    """插入定位依赖的两侧邻居均已被删除，插入也不能丢失。"""
    a, c = TextReplica("A"), TextReplica("C")
    m_xy = a.insert(0, "XY")
    m_del = a.delete(0, 2)            # X、Y 都成墓碑
    assert a.text() == ""
    m_z = a.insert(0, "z")            # 在墓碑 X 与 Y 之间插入
    assert a.text() == "z"
    # C 乱序接收：先插入、再删除、最后 z
    _deliver(c, m_xy)
    _deliver(c, m_del)
    _deliver(c, m_z)
    assert c.text() == "z"
    # 另一种顺序：z 先于删除到达
    c2 = TextReplica("C")
    _deliver(c2, m_z)                 # z 的邻居未知，缓冲
    _deliver(c2, m_del)               # 删除先到，缓冲
    _deliver(c2, m_xy)                # 依赖补齐
    assert c2.text() == "z"


def test_independent_updates_progress_while_dependency_missing():
    """缺失依赖不阻塞其他更新；只收到部分消息时副本可继续本地编辑。"""
    a, b = TextReplica("A"), TextReplica("B")
    m1 = a.insert(0, "a")             # b 依赖 a
    m2 = a.insert(1, "b")
    _deliver(b, m2)                   # 'b' 的父节点 'a' 缺失：仅缓冲 'b'
    assert b.text() == ""
    m_local = b.insert(0, "z")        # 本地编辑不受影响
    assert b.text() == "z"
    _deliver(a, m_local)
    assert a.text() == "abz" or a.text() == "zab"  # 不阻塞 A
    _deliver(b, m1)                   # 依赖补齐，'b' 落库
    assert b.text() == "abz"          # A<B：a、b 片段在前
    assert a.text() == "abz"


def test_delete_only_affects_visible_range():
    """删除只影响发起时可见的字符，不吞掉未见过的并发插入。"""
    a, b = _synced_pair()             # "LR"
    m_ins = b.insert(1, "m")          # B 在 L、R 间并发插入，A 未见过
    m_del = a.delete(0, 2)            # A 删除自己可见的 "LR"
    assert a.text() == ""
    _deliver(a, m_ins)
    _deliver(b, m_del)
    assert a.text() == b.text() == "m"  # 并发插入的 m 存活


def test_three_replicas_random_interleaved_convergence():
    """固定随机种子的三副本交错编辑：重复字符 + 非 BMP 字符，
    部分合流后继续编辑，最终补齐全部消息（含重复投递）后收敛。
    每次本地操作都用字符串切片校验即时结果。
    """
    rng = random.Random(20261005)
    replicas = [TextReplica(rid) for rid in ("r1", "r2", "r3")]
    alphabet = ["a", "b", "a", "🙂", "𝄞", "🙂", "中", "a", "b"]  # 含重复与非 BMP
    inboxes = {rid: [] for rid in ("r1", "r2", "r3")}            # 待投递消息

    def broadcast(sender_idx, msg):
        for j, r in enumerate(replicas):
            if j != sender_idx:
                inboxes[r.replica_id].append(msg)
                if rng.random() < 0.3:                            # 重复投递
                    inboxes[r.replica_id].append(msg)

    def deliver_some(prob):
        for r in replicas:
            pending = inboxes[r.replica_id]
            rng.shuffle(pending)                                  # 乱序投递
            keep = []
            for msg in pending:
                if rng.random() < prob:
                    r.receive(msg)
                else:
                    keep.append(msg)
            inboxes[r.replica_id] = keep

    total_ops = 240
    for step in range(total_ops):
        i = rng.randrange(3)
        r = replicas[i]
        before = r.text()
        if before and rng.random() < 0.35:
            start = rng.randrange(len(before))
            end = rng.randrange(start + 1, len(before) + 1)
            msg = r.delete(start, end)
            assert r.text() == before[:start] + before[end:]      # 切片校验
        else:
            pos = rng.randrange(len(before) + 1)
            n = rng.randint(1, 4)
            s = "".join(rng.choice(alphabet) for _ in range(n))
            msg = r.insert(pos, s)
            assert r.text() == before[:pos] + s + before[pos:]    # 切片校验
        broadcast(i, msg)
        # 前 60% 的操作只随机投递一部分（部分合流），之后继续编辑
        deliver_some(0.5 if step < total_ops * 0.6 else 0.8)

    # 最终补齐全部消息，并再重复投递一遍
    for r in replicas:
        for msg in inboxes[r.replica_id]:
            r.receive(msg)
            r.receive(msg)
        inboxes[r.replica_id] = []

    texts = [r.text() for r in replicas]
    assert texts[0] == texts[1] == texts[2], texts
    assert texts[0] != ""  #  Sanity：确实有内容参与合流


def test_messages_are_json_serializable():
    """消息必须可 JSON 序列化，且经 JSON 往返后 receive 仍正常工作。"""
    import json
    a, b = TextReplica("A"), TextReplica("B")
    m1 = a.insert(0, "hi🙂")
    m2 = a.delete(0, 1)
    for m in (m1, m2):
        b.receive(json.loads(json.dumps(m, ensure_ascii=False)))
    assert a.text() == b.text() == "i🙂"
