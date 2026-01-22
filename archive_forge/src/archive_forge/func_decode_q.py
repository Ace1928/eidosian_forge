import re
import base64
import binascii
import functools
from string import ascii_letters, digits
from email import errors
def decode_q(encoded):
    encoded = encoded.replace(b'_', b' ')
    return (_q_byte_subber(encoded), [])