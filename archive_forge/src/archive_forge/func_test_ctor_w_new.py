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
def test_ctor_w_new(self):
    """constructor -- 'new'  parameter"""
    self.assertRaises(TypeError, TOTP)
    self.assertRaises(TypeError, TOTP, key='4aoggdbbqsyhntuz', new=True)
    otp = TOTP(new=True)
    otp2 = TOTP(new=True)
    self.assertNotEqual(otp.key, otp2.key)