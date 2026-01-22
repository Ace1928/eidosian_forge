import re
import base64
import binascii
import functools
from string import ascii_letters, digits
from email import errors
def encode_q(bstring):
    return ''.join((_q_byte_map[x] for x in bstring))