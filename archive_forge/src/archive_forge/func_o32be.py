from __future__ import annotations
from struct import pack, unpack_from
def o32be(i: int) -> bytes:
    return pack('>I', i)