import struct
import sys
import numpy as np
def encode_buffer(l, infos=None):
    """Encode a list of arrays into a single byte array."""
    if not isinstance(l, list):
        raise ValueError('requires list')
    return encode_chunks(encode_list(l, infos=infos))