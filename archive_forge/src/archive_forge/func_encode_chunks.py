import struct
import sys
import numpy as np
def encode_chunks(l):
    """Encode a list of chunks into a single byte array, with lengths and magics.."""
    size = sum((16 + roundup(b.nbytes) for b in l))
    result = bytearray(size)
    offset = 0
    for b in l:
        result[offset:offset + 8] = magic_bytes
        offset += 8
        result[offset:offset + 8] = struct.pack('@q', b.nbytes)
        offset += 8
        result[offset:offset + bytelen(b)] = b
        offset += roundup(bytelen(b))
    return result