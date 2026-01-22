import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import sys
import time as _time
from passlib import exc
from passlib.utils.compat import unicode, u
from passlib.tests.utils import TestCase, time_call
from passlib import totp as totp_module
from passlib.totp import TOTP, AppWallet, AES_SUPPORT
def test_ctor_w_size(self):
    """constructor -- 'size'  parameter"""
    self.assertEqual(len(TOTP(new=True, alg='sha1').key), 20)
    self.assertEqual(len(TOTP(new=True, alg='sha256').key), 32)
    self.assertEqual(len(TOTP(new=True, alg='sha512').key), 64)
    self.assertEqual(len(TOTP(new=True, size=10).key), 10)
    self.assertEqual(len(TOTP(new=True, size=16).key), 16)
    self.assertRaises(ValueError, TOTP, new=True, size=21, alg='sha1')
    self.assertRaises(ValueError, TOTP, new=True, size=9)
    with self.assertWarningList([dict(category=exc.PasslibSecurityWarning, message_re='.*for security purposes, secret key must be.*')]):
        _ = TOTP('0A' * 9, 'hex')