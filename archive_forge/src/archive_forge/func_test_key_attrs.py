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
def test_key_attrs(self):
    """pretty_key() and .key attributes"""
    rng = self.getRandom()
    otp = TOTP(KEY1_RAW, 'raw')
    self.assertEqual(otp.key, KEY1_RAW)
    self.assertEqual(otp.hex_key, 'e01c630c2184b076ce99')
    self.assertEqual(otp.base32_key, KEY1)
    self.assertEqual(otp.pretty_key(), '4AOG-GDBB-QSYH-NTUZ')
    self.assertEqual(otp.pretty_key(sep=' '), '4AOG GDBB QSYH NTUZ')
    self.assertEqual(otp.pretty_key(sep=False), KEY1)
    self.assertEqual(otp.pretty_key(format='hex'), 'e01c-630c-2184-b076-ce99')
    otp = TOTP(new=True, size=rng.randint(10, 20))
    _ = otp.hex_key
    _ = otp.base32_key
    _ = otp.pretty_key()