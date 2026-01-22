from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
def test_needs_update_w_even_rounds(self):
    """needs_update() should flag even rounds"""
    handler = self.handler
    even_hash = '_Y/../cG0zkJa6LY6k4c'
    odd_hash = '_Z/..TgFg0/ptQtpAgws'
    secret = 'test'
    self.assertTrue(handler.verify(secret, even_hash))
    self.assertTrue(handler.verify(secret, odd_hash))
    self.assertTrue(handler.needs_update(even_hash))
    self.assertFalse(handler.needs_update(odd_hash))
    new_hash = handler.hash('stub')
    self.assertFalse(handler.needs_update(new_hash))