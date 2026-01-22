import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_93_derive_digest(self):
    """test scram.derive_digest()"""
    hash = self.handler.derive_digest
    s1 = b'\x01\x02\x03'
    d1 = b'\xb2\xfb\xab\x82[tNuPnI\x8aZZ\x19\x87\xcen\xe9\xd3'
    self.assertEqual(hash(u('Ⅸ'), s1, 1000, 'sha-1'), d1)
    self.assertEqual(hash(b'\xe2\x85\xa8', s1, 1000, 'SHA-1'), d1)
    self.assertEqual(hash(u('IX'), s1, 1000, 'sha1'), d1)
    self.assertEqual(hash(b'IX', s1, 1000, 'SHA1'), d1)
    self.assertEqual(hash('IX', s1, 1000, 'md5'), b'3\x19\x18\xc0\x1c/\xa8\xbf\xe4\xa3\xc2\x8eM\xe8od')
    self.assertRaises(ValueError, hash, 'IX', s1, 1000, 'sha-666')
    self.assertRaises(ValueError, hash, 'IX', s1, 0, 'sha-1')
    self.assertEqual(hash(u('IX'), s1.decode('latin-1'), 1000, 'sha1'), d1)