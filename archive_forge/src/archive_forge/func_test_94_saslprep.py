import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_94_saslprep(self):
    """test hash/verify use saslprep"""
    h = self.do_encrypt(u('I\xadX'))
    self.assertTrue(self.do_verify(u('IX'), h))
    self.assertTrue(self.do_verify(u('Ⅸ'), h))
    h = self.do_encrypt(u('ó'))
    self.assertTrue(self.do_verify(u('ó'), h))
    self.assertTrue(self.do_verify(u('\u200dó'), h))
    self.assertRaises(ValueError, self.do_encrypt, u('\ufdd0'))
    self.assertRaises(ValueError, self.do_verify, u('\ufdd0'), h)