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
def test_n_param(self):
    """'n' (rounds) parameter"""

    def run_scrypt(n):
        return hexstr(scrypt_mod.scrypt('secret', 'salt', n, 2, 2, 16))
    self.assertRaises(ValueError, run_scrypt, -1)
    self.assertRaises(ValueError, run_scrypt, 0)
    self.assertRaises(ValueError, run_scrypt, 1)
    self.assertEqual(run_scrypt(2), 'dacf2bca255e2870e6636fa8c8957a66')
    self.assertRaises(ValueError, run_scrypt, 3)
    self.assertRaises(ValueError, run_scrypt, 15)
    self.assertEqual(run_scrypt(16), '0272b8fc72bc54b1159340ed99425233')