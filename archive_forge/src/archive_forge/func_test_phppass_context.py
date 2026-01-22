from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_phppass_context(self):
    ctx = apps.phpass_context
    for hash in ['$P$8Ja1vJsKa5qyy/b3mCJGXM7GyBnt6..', '$H$8b95CoYQnQ9Y6fSTsACyphNh5yoM02.', '_cD..aBxeRhYFJvtUvsI']:
        self.assertTrue(ctx.verify('test', hash))
    h1 = '$2a$04$yjDgE74RJkeqC0/1NheSSOrvKeu9IbKDpcQf/Ox3qsrRS/Kw42qIS'
    if hashmod.bcrypt.has_backend():
        self.assertTrue(ctx.verify('test', h1))
        self.assertEqual(ctx.default_scheme(), 'bcrypt')
        self.assertEqual(ctx.handler().name, 'bcrypt')
    else:
        self.assertEqual(ctx.identify(h1), 'bcrypt')
        self.assertEqual(ctx.default_scheme(), 'phpass')
        self.assertEqual(ctx.handler().name, 'phpass')