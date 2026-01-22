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
def check_bmix(r, input, output):
    """helper to check bmix() output against reference"""
    engine = ScryptEngine(r=r, n=1 << rng.randint(1, 32), p=rng.randint(1, 1023))
    target = [rng.randint(0, 1 << 32) for _ in range(2 * r * 16)]
    engine.bmix(input, target)
    self.assertEqual(target, list(output))
    if r == 1:
        del engine.bmix
        target = [rng.randint(0, 1 << 32) for _ in range(2 * r * 16)]
        engine.bmix(input, target)
        self.assertEqual(target, list(output))