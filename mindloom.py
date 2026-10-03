"""MindLoom 单页协同文本内核（纯文本、内存态、最多 12 个副本）。

设计要点（详见 README.md）：
- 每个字符有全局唯一 id：(replica_id, seq)，seq 为副本内递增计数。
- 字符按“锚点树”组织：每个字符挂在某个左邻居（anchor）之下，anchor=None 表示文首。
- 同一锚点下的子节点排序键为 (group, replica_id, within)：
  group 大者靠前；group 相同比 replica_id（大者靠前）；再相同比 within（小者靠前）。
- 同一副本在同一缝隙连续前插/后插形成的“片段”共享一个 group，保证合流后不交错。
- 删除即墓碑：delete 消息只携带目标字符 id 列表。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

MAX_REPLICAS = 12

CharId = Tuple[Any, int]
Message = Dict[str, Any]

_ROOT: Optional[CharId] = None


@dataclass
class _Char:
    char_id: CharId
    value: str
    anchor: Optional[CharId]
    group: int
    within: int
    deleted: bool = False


def _sort_key(char: _Char) -> Tuple:
    # group 降序、replica_id 降序、within 升序
    return (-char.group, _DescKey(char.char_id[0]), char.within)


class _DescKey:
    """包装可比较值，翻转比较方向（用于 replica_id 降序）。"""

    __slots__ = ("v",)

    def __init__(self, v: Any) -> None:
        self.v = v

    def __lt__(self, other: "_DescKey") -> bool:
        return other.v < self.v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DescKey) and other.v == self.v


class Replica:
    """一个协同副本。replica_id 唯一且固定，初始内容为空。"""

    def __init__(self, replica_id: Any) -> None:
        self.replica_id = replica_id
        self._seq = 0
        self._chars: Dict[CharId, _Char] = {}
        # anchor_id（None 表示文首）-> 已排序的子字符 id 列表
        self._children: Dict[Optional[CharId], List[CharId]] = {_ROOT: []}
        # 已收到删除但尚未见到创建的目标 id
        self._deleted_ids: set = set()
        # 依赖未满足的插入记录（缓冲）
        self._buffer: List[dict] = []

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _next_id(self) -> CharId:
        self._seq += 1
        return (self.replica_id, self._seq)

    def _add_char(self, char: _Char) -> None:
        self._chars[char.char_id] = char
        siblings = self._children.setdefault(char.anchor, [])
        keys = [_sort_key(self._chars[c]) for c in siblings]
        idx = bisect.bisect_left(keys, _sort_key(char))
        siblings.insert(idx, char.char_id)
        self._children.setdefault(char.char_id, [])

    def _visible_ids(self) -> List[CharId]:
        out: List[CharId] = []
        stack: List[CharId] = list(reversed(self._children[_ROOT]))
        while stack:
            cid = stack.pop()
            char = self._chars[cid]
            if not char.deleted:
                out.append(cid)
            stack.extend(reversed(self._children[cid]))
        return out

    # ------------------------------------------------------------------
    # 本地操作
    # ------------------------------------------------------------------

    def insert(self, pos: int, text: str) -> Message:
        """在当前可见文本 pos 处插入 text，立即本地生效并返回消息。"""
        records: List[dict] = []
        for offset, ch in enumerate(text):
            visible = self._visible_ids()
            anchor = visible[pos + offset - 1] if pos + offset > 0 else None
            right = visible[pos + offset] if pos + offset < len(visible) else None
            group, within = self._alloc_key(anchor, right)
            char = _Char(
                char_id=self._next_id(),
                value=ch,
                anchor=anchor,
                group=group,
                within=within,
            )
            self._add_char(char)
            records.append(self._record_of(char))
        return {"type": "insert", "chars": records}

    def _alloc_key(self, anchor: Optional[CharId], right: Optional[CharId]) -> Tuple[int, int]:
        """为新字符分配 (group, within)。

        若右邻居与本新字符同锚点且同属本副本，则视为“向左前插”，
        并入右邻居所在 group 并取更小的 within，保证片段不交错；
        否则开启新 group，其值大于该锚点下所有已知 group。
        """
        siblings = self._children.get(anchor, [])
        if right is not None:
            right_char = self._chars[right]
            if right_char.anchor == anchor and right_char.char_id[0] == self.replica_id:
                min_within = min(
                    (self._chars[c].within for c in siblings
                     if self._chars[c].group == right_char.group
                     and c[0] == self.replica_id),
                    default=0,
                )
                return right_char.group, min_within - 1
        max_group = max((self._chars[c].group for c in siblings), default=0)
        return max_group + 1, 0

    @staticmethod
    def _record_of(char: _Char) -> dict:
        return {
            "id": [char.char_id[0], char.char_id[1]],
            "anchor": list(char.anchor) if char.anchor is not None else None,
            "group": char.group,
            "within": char.within,
            "ch": char.value,
        }

    def delete(self, start: int, end: int) -> Message:
        """删除可见文本左闭右开区间 [start, end)，立即本地生效并返回消息。"""
        visible = self._visible_ids()
        targets = visible[start:end]
        for cid in targets:
            self._chars[cid].deleted = True
            self._deleted_ids.add(cid)
        return {"type": "delete", "ids": [[c[0], c[1]] for c in targets]}

    # ------------------------------------------------------------------
    # 消息接收
    # ------------------------------------------------------------------

    def receive(self, message: Message) -> None:
        """接收消息（可重复、乱序）；不等待缺失消息。

        依赖未满足的插入记录进入缓冲，其余消息照常生效；
        每次成功应用缓冲记录后重放剩余缓冲，直到没有进展。
        """
        if message["type"] == "insert":
            self._buffer.extend(message["chars"])
            self._drain_buffer()
        elif message["type"] == "delete":
            for raw in message["ids"]:
                cid: CharId = (raw[0], raw[1])
                self._deleted_ids.add(cid)
                char = self._chars.get(cid)
                if char is not None:
                    char.deleted = True
        else:
            raise ValueError(f"unknown message type: {message['type']!r}")

    def _drain_buffer(self) -> None:
        progress = True
        while progress:
            progress = False
            remaining: List[dict] = []
            for rec in self._buffer:
                if self._apply_record(rec):
                    progress = True
                else:
                    remaining.append(rec)
            self._buffer = remaining

    def _apply_record(self, rec: dict) -> bool:
        cid: CharId = (rec["id"][0], rec["id"][1])
        if cid in self._chars:
            return True  # 重复投递：幂等忽略
        anchor: Optional[CharId] = tuple(rec["anchor"]) if rec["anchor"] is not None else None
        if anchor is not None and anchor not in self._chars:
            return False  # 锚点尚未到达，继续缓冲
        char = _Char(
            char_id=cid,
            value=rec["ch"],
            anchor=anchor,
            group=rec["group"],
            within=rec["within"],
            deleted=cid in self._deleted_ids,
        )
        self._add_char(char)
        return True

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def text(self) -> str:
        """返回当前可见文本。"""
        return "".join(self._chars[c].value for c in self._visible_ids())
