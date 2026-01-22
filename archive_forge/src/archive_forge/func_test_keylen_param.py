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
def test_keylen_param(self):
    """'keylen' parameter"""
    rng = self.getRandom()

    def run_scrypt(keylen):
        return hexstr(scrypt_mod.scrypt('secret', 'salt', 2, 2, 2, keylen))
    self.assertRaises(ValueError, run_scrypt, -1)
    self.assertRaises(ValueError, run_scrypt, 0)
    self.assertEqual(run_scrypt(1), 'da')
    ksize = rng.randint(1, 1 << 10)
    self.assertEqual(len(run_scrypt(ksize)), 2 * ksize)
    self.assertRaises(ValueError, run_scrypt, (2 ** 32 - 1) * 32 + 1)