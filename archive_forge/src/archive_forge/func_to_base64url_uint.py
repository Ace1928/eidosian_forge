import base64
import binascii
import re
from typing import Union
def to_base64url_uint(val: int) -> bytes:
    if val < 0:
        raise ValueError('Must be a positive integer')
    int_bytes = bytes_from_int(val)
    if len(int_bytes) == 0:
        int_bytes = b'\x00'
    return base64url_encode(int_bytes)