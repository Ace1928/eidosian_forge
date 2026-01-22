from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
def test_90_raw(self):
    """test lmhash.raw() method"""
    from binascii import unhexlify
    from passlib.utils.compat import str_to_bascii
    lmhash = self.handler
    for secret, hash in self.known_correct_hashes:
        kwds = {}
        secret = self.populate_context(secret, kwds)
        data = unhexlify(str_to_bascii(hash))
        self.assertEqual(lmhash.raw(secret, **kwds), data)
    self.assertRaises(TypeError, lmhash.raw, 1)