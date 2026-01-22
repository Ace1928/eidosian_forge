import unittest
import binascii
from Cryptodome.Util.py3compat import b
def b2a_hex(s):
    """Convert binary to hexadecimal"""
    return binascii.b2a_hex(s)