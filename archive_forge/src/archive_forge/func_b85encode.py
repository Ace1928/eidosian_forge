import re
import struct
import binascii
def b85encode(b, pad=False):
    """Encode bytes-like object b in base85 format and return a bytes object.

    If pad is true, the input is padded with b'\\0' so its length is a multiple of
    4 bytes before encoding.
    """
    global _b85chars, _b85chars2
    if _b85chars2 is None:
        _b85chars = [bytes((i,)) for i in _b85alphabet]
        _b85chars2 = [a + b for a in _b85chars for b in _b85chars]
    return _85encode(b, _b85chars, _b85chars2, pad)