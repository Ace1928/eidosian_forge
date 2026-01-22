import unittest
from Cryptodome.Util.py3compat import b
from Cryptodome.SelfTest.st_common import list_test_cases
from binascii import unhexlify
from Cryptodome.Cipher import ARC4
def test_keystream(self):
    for tv in self.rfc6229_data:
        key = unhexlify(b(tv[0]))
        cipher = ARC4.new(key)
        count = 0
        for offset in range(0, 4096 + 1, 16):
            ct = cipher.encrypt(b('\x00') * 16)
            expected = tv[1].get(offset)
            if expected:
                expected = unhexlify(b(expected.replace(' ', '')))
                self.assertEqual(ct, expected)
                count += 1
        self.assertEqual(count, len(tv[1]))