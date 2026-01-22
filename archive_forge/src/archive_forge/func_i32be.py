from __future__ import annotations
from struct import pack, unpack_from
def i32be(c: bytes, o: int=0) -> int:
    return unpack_from('>I', c, o)[0]