from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_django16_context(self):
    ctx = apps.django16_context
    for hash in ['pbkdf2_sha256$29000$ZsgquwnCyBs2$fBxRQpfKd2PIeMxtkKPy0h7SrnrN+EU/cm67aitoZ2s=', 'sha1$0d082$cdb462ae8b6be8784ef24b20778c4d0c82d5957f', 'md5$b887a$37767f8a745af10612ad44c80ff52e92', 'crypt$95a6d$95x74hLDQKXI2', '098f6bcd4621d373cade4e832627b4f6']:
        self.assertTrue(ctx.verify('test', hash))
    self.assertEqual(ctx.identify('!'), 'django_disabled')
    self.assertFalse(ctx.verify('test', '!'))