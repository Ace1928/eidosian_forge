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
def test_totp_token(self):
    """generate() -- TotpToken() class"""
    from passlib.totp import TOTP, TotpToken
    otp = TOTP('s3jdvb7qd2r7jpxx')
    result = otp.generate(1419622739)
    self.assertIsInstance(result, TotpToken)
    self.assertEqual(result.token, '897212')
    self.assertEqual(result.counter, 47320757)
    self.assertEqual(result.expire_time, 1419622740)
    self.assertEqual(result, ('897212', 1419622740))
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0], '897212')
    self.assertEqual(result[1], 1419622740)
    self.assertRaises(IndexError, result.__getitem__, -3)
    self.assertRaises(IndexError, result.__getitem__, 2)
    self.assertTrue(result)
    otp.now = lambda: 1419622739.5
    self.assertEqual(result.remaining, 0.5)
    self.assertTrue(result.valid)
    otp.now = lambda: 1419622741
    self.assertEqual(result.remaining, 0)
    self.assertFalse(result.valid)
    result2 = otp.generate(1419622739)
    self.assertIsNot(result2, result)
    self.assertEqual(result2, result)
    result3 = otp.generate(1419622711)
    self.assertIsNot(result3, result)
    self.assertEqual(result3, result)
    result4 = otp.generate(1419622999)
    self.assertNotEqual(result4, result)