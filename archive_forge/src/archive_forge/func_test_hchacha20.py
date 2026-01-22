import os
import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import ChaCha20
def test_hchacha20(self):
    from Cryptodome.Cipher.ChaCha20 import _HChaCha20
    key = b'00:01:02:03:04:05:06:07:08:09:0a:0b:0c:0d:0e:0f:10:11:12:13:14:15:16:17:18:19:1a:1b:1c:1d:1e:1f'
    key = unhexlify(key.replace(b':', b''))
    nonce = b'00:00:00:09:00:00:00:4a:00:00:00:00:31:41:59:27'
    nonce = unhexlify(nonce.replace(b':', b''))
    subkey = _HChaCha20(key, nonce)
    expected = b'82413b42 27b27bfe d30e4250 8a877d73 a0f9e4d5 8a74a853 c12ec413 26d3ecdc'
    expected = unhexlify(expected.replace(b' ', b''))
    self.assertEqual(subkey, expected)