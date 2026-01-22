from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import hosts, hash as hashmod
from passlib.utils import unix_crypt_schemes
from passlib.tests.utils import TestCase
def test_host_context(self):
    ctx = getattr(hosts, 'host_context', None)
    if not ctx:
        return self.skipTest('host_context not available on this platform')
    schemes = list(ctx.schemes())
    self.assertTrue(schemes, 'appears to be unix system, but no known schemes supported by crypt')
    self.assertTrue('unix_disabled' in schemes)
    schemes.remove('unix_disabled')
    self.assertTrue(schemes, 'should have schemes beside fallback scheme')
    self.assertTrue(set(unix_crypt_schemes).issuperset(schemes))
    self.check_unix_disabled(ctx)
    for scheme, hash in [('sha512_crypt', '$6$rounds=41128$VoQLvDjkaZ6L6BIE$4pt.1Ll1XdDYduEwEYPCMOBiR6W6znsyUEoNlcVXpv2gKKIbQolgmTGe6uEEVJ7azUxuc8Tf7zV9SD2z7Ij751'), ('sha256_crypt', '$5$rounds=31817$iZGmlyBQ99JSB5n6$p4E.pdPBWx19OajgjLRiOW0itGnyxDGgMlDcOsfaI17'), ('md5_crypt', '$1$TXl/FX/U$BZge.lr.ux6ekjEjxmzwz0'), ('des_crypt', 'kAJJz.Rwp0A/I')]:
        if scheme in schemes:
            self.assertTrue(ctx.verify('test', hash))