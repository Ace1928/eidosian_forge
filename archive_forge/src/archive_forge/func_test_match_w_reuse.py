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
def test_match_w_reuse(self):
    """match() -- 'reuse' and 'last_counter' parameters"""
    otp = self.randotp()
    period = otp.period
    time = self.randtime()
    tdata = otp.generate(time)
    token = tdata.token
    counter = tdata.counter
    expire_time = tdata.expire_time
    common = dict(otp=otp, gen_time=time)
    assertMatches = partial(self.assertVerifyMatches, **common)
    assertRaises = partial(self.assertVerifyRaises, **common)
    assertMatches(-1, token, time + period, window=period)
    assertMatches(-1, token, time + period, last_counter=counter - 1, window=period)
    assertRaises(exc.InvalidTokenError, token, time + 2 * period, last_counter=counter, window=period)
    err = assertRaises(exc.UsedTokenError, token, time + period, last_counter=counter, window=period)
    self.assertEqual(err.expire_time, expire_time)
    err = assertRaises(exc.UsedTokenError, token, time, last_counter=counter, window=0)
    self.assertEqual(err.expire_time, expire_time)