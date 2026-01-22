import struct
import sys
import numpy as np
def decode_chunks(buf):
    """Decode a byte array into a list of chunks."""
    result = []
    offset = 0
    total = bytelen(buf)
    while offset < total:
        if magic_bytes != buf[offset:offset + 8]:
            raise ValueError('magic bytes mismatch')
        offset += 8
        nbytes = struct.unpack('@q', buf[offset:offset + 8])[0]
        offset += 8
        b = buf[offset:offset + nbytes]
        offset += roundup(nbytes)
        result.append(b)
    return result