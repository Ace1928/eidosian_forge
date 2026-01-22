import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_94_needs_update_algs(self):
    """needs_update() -- algs setting"""
    handler1 = self.handler.using(algs='sha1,md5')
    h1 = handler1.hash('dummy')
    self.assertFalse(handler1.needs_update(h1))
    handler2 = handler1.using(algs='sha1')
    self.assertFalse(handler2.needs_update(h1))
    handler3 = handler1.using(algs='sha1,sha256')
    self.assertTrue(handler3.needs_update(h1))