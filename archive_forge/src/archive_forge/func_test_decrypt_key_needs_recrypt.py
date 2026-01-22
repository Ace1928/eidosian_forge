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
def test_decrypt_key_needs_recrypt(self):
    """.decrypt_key() -- needs_recrypt flag"""
    self.require_aes_support()
    wallet = AppWallet({'1': PASS1, '2': PASS2}, encrypt_cost=13)
    ref = dict(v=1, c=13, s='AAAA', k='AAAA', t='2')
    self.assertFalse(wallet.decrypt_key(ref)[1])
    temp = ref.copy()
    temp.update(c=8)
    self.assertTrue(wallet.decrypt_key(temp)[1])
    temp = ref.copy()
    temp.update(t='1')
    self.assertTrue(wallet.decrypt_key(temp)[1])