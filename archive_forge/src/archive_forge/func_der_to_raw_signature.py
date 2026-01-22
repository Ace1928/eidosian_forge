import base64
import binascii
import re
from typing import Union
def der_to_raw_signature(der_sig: bytes, curve: 'EllipticCurve') -> bytes:
    num_bits = curve.key_size
    num_bytes = (num_bits + 7) // 8
    r, s = decode_dss_signature(der_sig)
    return number_to_bytes(r, num_bytes) + number_to_bytes(s, num_bytes)