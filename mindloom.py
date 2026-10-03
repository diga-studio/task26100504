"""MindLoom 单页协同文本内核（纯文本 CRDT）。

- 纯文本、纯内存、单线程，最多 12 个副本，replica_id 唯一且固定。
- 字符级 CRDT：每个字符有全局唯一 id (replica_id, 序号)，删除即墓碑。
- 消息可重复、乱序、延迟；所有副本收齐同一批消息后文本逐码点收敛。
- 位置一律按 Unicode 码点计数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 字符唯一标识：(replica_id, 序号)。序号从 1 开始，按字符递增。
CharId = Tuple[Any, int]

# 可 JSON 序列化的消息。
Message = Dict[str, Any]

@dataclass
class _Node:
    """一个字符节点（含墓碑）。"""

    char: str
    parent: Optional[CharId]      # 插入时左侧相邻节点（遍历序，含墓碑）；None 表示文档起点
    right: Optional[CharId]       # 插入时右侧相邻节点；None 表示文档末尾
    deleted: bool = False


class TextReplica:
    """一个文本副本。replica_id 唯一且固定，初始内容为空。"""

    def __init__(self, replica_id: Any) -> None:
        self.replica_id = replica_id
        self._seq = 0
        self._nodes: Dict[CharId, _Node] = {}
        self._pending_inserts: List[Tuple[CharId, _Node]] = []  # 依赖缺失的孤儿插入
        self._pending_deletes: List[CharId] = []                # 目标尚未到达的删除

    # ---- 内部：排序与遍历 ----

    def _ordered_children(self, parent: Optional[CharId]) -> List[CharId]:
        """计算某节点的子节点稳定顺序。

        排序规则（只依赖消息携带的静态元数据，与到达顺序无关）：
        1. 右邻也是同父子节点的，通过 right 链挂到该右邻前面，形成“链”；
           指向同一右邻的多个节点按 (replica_id, 序号降序) 排列，各自先递归
           输出自己的前驱子链，再输出自身。
        2. 右邻不是同父子节点的节点是链尾；每条链对应一个 replica（同一
           replica 在同一父节点下只会有一条链尾），链之间按链尾的
           (replica_id, 序号) 升序排列。
        效果：同一 replica 从同一缝隙连续前插/追加的字符构成连续片段，
        片段之间按 replica_id 升序排列；后到的插入若指明右邻为已有子
        节点，则恰好落在该右邻之前，保证本地即时结果与字符串切片一致。
        """
        kids = [cid for cid, n in self._nodes.items() if n.parent == parent]
        kidset = set(kids)
        preds_of: Dict[CharId, List[CharId]] = {}
        tails: List[CharId] = []
        for cid in kids:
            right = self._nodes[cid].right
            if right in kidset:
                preds_of.setdefault(right, []).append(cid)
            else:
                tails.append(cid)
        tails.sort(key=lambda c: (c[0], c[1]))

        ordered: List[CharId] = []

        def emit_chain(tail: CharId) -> None:
            preds = preds_of.get(tail)
            if preds:
                preds.sort(key=lambda c: (c[0], -c[1]))
                for pred in preds:
                    emit_chain(pred)
            ordered.append(tail)

        for tail in tails:
            emit_chain(tail)
        return ordered

    def _traversal(self) -> List[CharId]:
        """全部节点（含墓碑）的稳定遍历序。"""
        order: List[CharId] = []

        def visit(cid: CharId) -> None:
            order.append(cid)
            for child in self._ordered_children(cid):
                visit(child)

        for child in self._ordered_children(None):
            visit(child)
        return order

    def insert(self, pos: int, text: str) -> Message:
        """在当前可见文本 pos 处插入 text，立即本地生效并返回消息。"""
        if not text:
            return {"op": "insert", "replica": self.replica_id, "seq": self._seq + 1,
                    "text": "", "parent": None, "right": None}
        order = self._traversal()
        # 找可见第 pos 个字符对应的遍历缝隙：取满足“之前可见字符数 == pos”的最大下标，
        # 即落在该位置所有墓碑之后，保证本地结果与字符串切片一致。
        visible = 0
        boundary = 0
        for idx, cid in enumerate(order):
            if visible == pos:
                boundary = idx
            if not self._nodes[cid].deleted:
                visible += 1
        if visible == pos:
            boundary = len(order)
        left = order[boundary - 1] if boundary > 0 else None
        right = order[boundary] if boundary < len(order) else None

        base = self._seq + 1
        parent = left
        for i, ch in enumerate(text):
            cid = (self.replica_id, base + i)
            # 只有首字符挂在消息的父节点下，其右邻（消息右侧节点）用于兄弟
            # 链排序；后续字符的右邻即下一字符（是它的子节点而非兄弟），对
            # 排序无意义，记 None 以避免依赖成环。
            node_right = right if i == 0 else None
            self._nodes[cid] = _Node(char=ch, parent=parent, right=node_right)
            parent = cid
        self._seq = base + len(text) - 1
        return {"op": "insert", "replica": self.replica_id, "seq": base,
                "text": text,
                "parent": list(left) if left is not None else None,
                "right": list(right) if right is not None else None}

    def delete(self, start: int, end: int) -> Message:
        """删除当前可见文本的 [start, end) 区间，立即本地生效并返回消息。"""
        targets: List[CharId] = []
        visible = 0
        for cid in self._traversal():
            node = self._nodes[cid]
            if node.deleted:
                continue
            if start <= visible < end:
                node.deleted = True
                targets.append(cid)
            visible += 1
        return {"op": "delete", "targets": [list(t) for t in targets]}

    def receive(self, message: Message) -> None:
        """接收消息；不等待缺失消息，依赖未满足的更新会被缓冲。"""
        if message["op"] == "insert":
            replica = message["replica"]
            base = message["seq"]
            chars = message["text"]
            parent = tuple(message["parent"]) if message["parent"] is not None else None
            right = tuple(message["right"]) if message["right"] is not None else None
            for i, ch in enumerate(chars):
                cid = (replica, base + i)
                node_right = right if i == 0 else None
                self._apply_insert(cid, _Node(char=ch, parent=parent, right=node_right))
                parent = cid
        elif message["op"] == "delete":
            for target in message["targets"]:
                self._apply_delete(tuple(target))
        else:
            raise ValueError(f"unknown op: {message['op']!r}")
        self._flush_pending()

    def _apply_insert(self, cid: CharId, node: _Node) -> None:
        # 重复投递幂等忽略：已落库或已在孤儿缓冲中都不能再次写入，
        # 否则重复的孤儿条目落库时会覆盖先到的删除标记。
        if cid in self._nodes or any(pid == cid for pid, _ in self._pending_inserts):
            return
        parent_known = node.parent is None or node.parent in self._nodes
        right_known = node.right is None or node.right in self._nodes
        if parent_known and right_known:
            self._nodes[cid] = node
            if cid in self._pending_deletes:
                node.deleted = True
                self._pending_deletes.remove(cid)
        else:
            self._pending_inserts.append((cid, node))

    def _apply_delete(self, target: CharId) -> None:
        node = self._nodes.get(target)
        if node is not None:
            node.deleted = True  # 幂等
        elif target not in self._pending_deletes:
            self._pending_deletes.append(target)  # 删除先于创建到达

    def _flush_pending(self) -> None:
        progressed = True
        while progressed:
            progressed = False
            remaining: List[Tuple[CharId, _Node]] = []
            for cid, node in self._pending_inserts:
                if cid in self._nodes:
                    progressed = True
                    continue
                parent_known = node.parent is None or node.parent in self._nodes
                right_known = node.right is None or node.right in self._nodes
                if parent_known and right_known:
                    self._nodes[cid] = node
                    if cid in self._pending_deletes:
                        node.deleted = True
                        self._pending_deletes.remove(cid)
                    progressed = True
                else:
                    remaining.append((cid, node))
            self._pending_inserts = remaining

    def text(self) -> str:
        """返回当前可见文本。"""
        return "".join(self._nodes[cid].char
                       for cid in self._traversal()
                       if not self._nodes[cid].deleted)
