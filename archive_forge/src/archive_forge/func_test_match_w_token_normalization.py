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
def test_match_w_token_normalization(self):
    """match() -- token normalization"""
    otp = TOTP('otxl2f5cctbprpzx')
    match = otp.match
    time = 1412889861
    self.assertTrue(match('    3 32-136  ', time))
    self.assertTrue(match(b'332136', time))
    self.assertRaises(exc.MalformedTokenError, match, '12345', time)
    self.assertRaises(exc.MalformedTokenError, match, '12345X', time)
    self.assertRaises(exc.MalformedTokenError, match, '0123456', time)