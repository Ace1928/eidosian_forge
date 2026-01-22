import unittest
from binascii import hexlify
from Cryptodome.Util.py3compat import tostr, tobytes
from Cryptodome.Hash import (HMAC, MD5, SHA1, SHA256,
def get_mv_rw(data):
    return memoryview(bytearray(data))