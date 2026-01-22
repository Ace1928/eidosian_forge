from __future__ import annotations
from struct import pack, unpack_from
def o16le(i: int) -> bytes:
    return pack('<H', i)