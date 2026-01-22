from __future__ import annotations
import json
import sqlite3
from itertools import zip_longest
from typing import Iterable
def num_in_numbits(num: int, numbits: bytes) -> bool:
    """Does the integer `num` appear in `numbits`?

    Returns:
        A bool, True if `num` is a member of `numbits`.
    """
    nbyte, nbit = divmod(num, 8)
    if nbyte >= len(numbits):
        return False
    return bool(numbits[nbyte] & 1 << nbit)