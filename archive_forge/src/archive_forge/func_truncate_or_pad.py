import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def truncate_or_pad(byte_string, size=None):
    if size is None:
        size = 32
    byte_array = bytearray(byte_string)
    length = len(byte_array)
    if length > size:
        return bytes(byte_array[:size])
    elif length < size:
        return bytes(byte_array + b'\x00' * (size - length))
    else:
        return byte_string