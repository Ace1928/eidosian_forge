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
def test_generate_w_reference_vectors(self):
    """generate() -- reference vectors"""
    for otp, time, token, expires, prefix in self.iter_test_vectors():
        result = otp.generate(time)
        self.assertEqual(result.token, token, msg=prefix)
        self.assertEqual(result.counter, time // otp.period, msg=prefix)
        if expires:
            self.assertEqual(result.expire_time, expires)