from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_master_context(self):
    ctx = apps.master_context
    self.assertGreater(len(ctx.schemes()), 50)