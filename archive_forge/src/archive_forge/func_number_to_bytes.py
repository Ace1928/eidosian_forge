import base64
import binascii
import re
from typing import Union
def number_to_bytes(num: int, num_bytes: int) -> bytes:
    padded_hex = '%0*x' % (2 * num_bytes, num)
    return binascii.a2b_hex(padded_hex.encode('ascii'))