from binascii import hexlify
import hashlib
import logging; log = logging.getLogger(__name__)
import struct
import warnings
from passlib import exc
from passlib.utils import getrandbytes
from passlib.utils.compat import PYPY, u, bascii_to_str
from passlib.utils.decor import classproperty
from passlib.tests.utils import TestCase, skipUnless, TEST_MODE, hb
from passlib.crypto import scrypt as scrypt_mod
def test_salsa(self):
    """salsa20()"""
    from passlib.crypto.scrypt._builtin import salsa20
    input = unpack_uint32_list(hb('\n            7e 87 9a 21 4f 3e c9 86 7c a9 40 e6 41 71 8f 26\n            ba ee 55 5b 8c 61 c1 b5 0d f8 46 11 6d cd 3b 1d\n            ee 24 f3 19 df 9b 3d 85 14 12 1e 4b 5a c5 aa 32\n            76 02 1d 29 09 c7 48 29 ed eb c6 8d b8 b8 c2 5e\n            '))
    output = unpack_uint32_list(hb('\n            a4 1f 85 9c 66 08 cc 99 3b 81 ca cb 02 0c ef 05\n            04 4b 21 81 a2 fd 33 7d fd 7b 1c 63 96 68 2f 29\n            b4 39 31 68 e3 c9 e6 bc fe 6b c5 b7 a0 6d 96 ba\n            e4 24 cc 10 2c 91 74 5c 24 ad 67 3d c7 61 8f 81\n            '))
    self.assertEqual(salsa20(input), output)
    input = list(range(16))
    output = unpack_uint32_list(hb('\n            f518dd4fb98883e0a87954c05cab867083bb8808552810752285a05822f56c16\n            9d4a2a0fd2142523d758c60b36411b682d53860514b871d27659042a5afa475d\n            '))
    self.assertEqual(salsa20(input), output)