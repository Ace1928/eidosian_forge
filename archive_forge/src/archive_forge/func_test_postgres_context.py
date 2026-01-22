from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_postgres_context(self):
    ctx = apps.postgres_context
    hash = 'md55d9c68c6c50ed3d02a2fcf54f63993b6'
    self.assertTrue(ctx.verify('test', hash, user='user'))