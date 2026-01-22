from __future__ import annotations
from functools import lru_cache
import collections
import enum
import os
import re
import typing as T
def extend_direct(self, iterable: T.Iterable[str]) -> None:
    """
        Extend using the elements in the specified iterable without any
        reordering or de-dup except for absolute paths where the order of
        include search directories is not relevant
        """
    self.flush_pre_post()
    for elem in iterable:
        self.append_direct(elem)