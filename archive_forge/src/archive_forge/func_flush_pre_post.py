from __future__ import annotations
from functools import lru_cache
import collections
import enum
import os
import re
import typing as T
def flush_pre_post(self) -> None:
    new: T.List[str] = []
    pre_flush_set: T.Set[str] = set()
    post_flush: T.Deque[str] = collections.deque()
    post_flush_set: T.Set[str] = set()
    for a in self.pre:
        dedup = self._can_dedup(a)
        if a not in pre_flush_set:
            new.append(a)
            if dedup is Dedup.OVERRIDDEN:
                pre_flush_set.add(a)
    for a in reversed(self.post):
        dedup = self._can_dedup(a)
        if a not in post_flush_set:
            post_flush.appendleft(a)
            if dedup is Dedup.OVERRIDDEN:
                post_flush_set.add(a)
    if pre_flush_set or post_flush_set:
        for a in self._container:
            if a not in post_flush_set and a not in pre_flush_set:
                new.append(a)
    else:
        new.extend(self._container)
    new.extend(post_flush)
    self._container = new
    self.pre.clear()
    self.post.clear()