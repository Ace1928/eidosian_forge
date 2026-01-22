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
def test_randotp(self):
    """
        internal test -- randotp()
        """
    otp1 = self.randotp()
    otp2 = self.randotp()
    self.assertNotEqual(otp1.key, otp2.key, 'key not randomized:')
    for _ in range(10):
        if otp1.digits != otp2.digits:
            break
        otp2 = self.randotp()
    else:
        self.fail('digits not randomized')
    for _ in range(10):
        if otp1.alg != otp2.alg:
            break
        otp2 = self.randotp()
    else:
        self.fail('alg not randomized')