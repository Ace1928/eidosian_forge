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
def test_from_uri(self):
    """from_uri()"""
    from passlib.totp import TOTP
    from_uri = TOTP.from_uri
    otp = from_uri('otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&issuer=Example')
    self.assertIsInstance(otp, TOTP)
    self.assertEqual(otp.key, KEY4_RAW)
    self.assertEqual(otp.label, 'alice@google.com')
    self.assertEqual(otp.issuer, 'Example')
    self.assertEqual(otp.alg, 'sha1')
    self.assertEqual(otp.period, 30)
    self.assertEqual(otp.digits, 6)
    otp = from_uri('otpauth://totp/Example:alice@google.com?secret=jbswy3dpehpk3pxp&issuer=Example')
    self.assertEqual(otp.key, KEY4_RAW)
    self.assertRaises(ValueError, from_uri, 'otpauth://totp/Example:alice@google.com?digits=6')
    self.assertRaises(Base32DecodeError, from_uri, 'otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHP@3PXP')
    otp = from_uri('otpauth://totp/Provider1:Alice%20Smith?secret=JBSWY3DPEHPK3PXP&issuer=Provider1')
    self.assertEqual(otp.label, 'Alice Smith')
    self.assertEqual(otp.issuer, 'Provider1')
    otp = from_uri('otpauth://totp/Big%20Corporation%3A%20alice@bigco.com?secret=JBSWY3DPEHPK3PXP')
    self.assertEqual(otp.label, 'alice@bigco.com')
    self.assertEqual(otp.issuer, 'Big Corporation')
    otp = from_uri('otpauth://totp/alice@bigco.com?secret=JBSWY3DPEHPK3PXP&issuer=Big%20Corporation')
    self.assertEqual(otp.label, 'alice@bigco.com')
    self.assertEqual(otp.issuer, 'Big Corporation')
    self.assertRaises(ValueError, TOTP.from_uri, 'otpauth://totp/Provider1:alice?secret=JBSWY3DPEHPK3PXP&issuer=Provider2')
    otp = from_uri('otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&algorithm=SHA256')
    self.assertEqual(otp.alg, 'sha256')
    self.assertRaises(ValueError, from_uri, 'otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&algorithm=SHA333')
    otp = from_uri('otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&digits=8')
    self.assertEqual(otp.digits, 8)
    self.assertRaises(ValueError, from_uri, 'otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&digits=A')
    self.assertRaises(ValueError, from_uri, 'otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&digits=%20')
    self.assertRaises(ValueError, from_uri, 'otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&digits=15')
    otp = from_uri('otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&period=63')
    self.assertEqual(otp.period, 63)
    self.assertRaises(ValueError, from_uri, 'otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&period=0')
    self.assertRaises(ValueError, from_uri, 'otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&period=-1')
    with self.assertWarningList([dict(category=exc.PasslibRuntimeWarning, message_re='unexpected parameters encountered')]):
        otp = from_uri('otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&foo=bar&period=63')
    self.assertEqual(otp.base32_key, KEY4)
    self.assertEqual(otp.period, 63)