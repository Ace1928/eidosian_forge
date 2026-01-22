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
def test_missing_backend(self):
    """backend management -- missing backend"""
    if has_stdlib_scrypt or has_cffi_scrypt:
        raise self.skipTest('non-builtin backend is present')
    self.assertRaises(exc.MissingBackendError, scrypt_mod._set_backend, 'scrypt')