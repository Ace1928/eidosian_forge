import re
import base64
import binascii
import functools
from string import ascii_letters, digits
from email import errors
def encode_b(bstring):
    return base64.b64encode(bstring).decode('ascii')