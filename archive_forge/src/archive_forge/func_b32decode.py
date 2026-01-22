import re
import struct
import binascii
def b32decode(s, casefold=False, map01=None):
    return _b32decode(_b32alphabet, s, casefold, map01)