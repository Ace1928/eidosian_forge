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
def test_match_w_skew(self):
    """match() -- 'skew' parameters"""
    otp = self.randotp()
    period = otp.period
    time = self.randtime()
    common = dict(otp=otp, gen_time=time)
    assertMatches = partial(self.assertVerifyMatches, **common)
    assertRaises = partial(self.assertVerifyRaises, **common)
    skew = 3 * period
    behind_token = otp.generate(time - skew).token
    assertRaises(exc.InvalidTokenError, behind_token, time, window=0)
    assertMatches(-3, behind_token, time, window=0, skew=-skew)
    ahead_token = otp.generate(time + skew).token
    assertRaises(exc.InvalidTokenError, ahead_token, time, window=0)
    assertMatches(+3, ahead_token, time, window=0, skew=skew)