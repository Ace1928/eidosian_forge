from __future__ import annotations
from struct import pack, unpack_from
def i8(c: bytes) -> int:
    return c[0]