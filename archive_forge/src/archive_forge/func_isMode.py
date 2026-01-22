import unittest
from binascii import a2b_hex, b2a_hex, hexlify
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.strxor import strxor_c
def isMode(self, name):
    if not hasattr(self.module, 'MODE_' + name):
        return False
    return self.mode == getattr(self.module, 'MODE_' + name)