from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_phpbb3_context(self):
    ctx = apps.phpbb3_context
    for hash in ['$P$8Ja1vJsKa5qyy/b3mCJGXM7GyBnt6..', '$H$8b95CoYQnQ9Y6fSTsACyphNh5yoM02.']:
        self.assertTrue(ctx.verify('test', hash))
    self.assertTrue(ctx.hash('test').startswith('$H$'))