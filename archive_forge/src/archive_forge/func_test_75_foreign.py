from __future__ import with_statement
from binascii import unhexlify
import contextlib
from functools import wraps, partial
import hashlib
import logging; log = logging.getLogger(__name__)
import random
import re
import os
import sys
import tempfile
import threading
import time
from passlib.exc import PasslibHashWarning, PasslibConfigWarning
from passlib.utils.compat import PY3, JYTHON
import warnings
from warnings import warn
from passlib import exc
from passlib.exc import MissingBackendError
import passlib.registry as registry
from passlib.tests.backports import TestCase as _TestCase, skip, skipIf, skipUnless, SkipTest
from passlib.utils import has_rounds_info, has_salt_info, rounds_cost_values, \
from passlib.utils.compat import iteritems, irange, u, unicode, PY2, nullcontext
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
def test_75_foreign(self):
    """test known foreign hashes"""
    if self.accepts_all_hashes:
        raise self.skipTest('not applicable')
    if not self.known_other_hashes:
        raise self.skipTest('no foreign hashes provided')
    for name, hash in self.known_other_hashes:
        if name == self.handler.name:
            self.assertTrue(self.do_identify(hash), 'identify() failed to identify known hash: %r' % (hash,))
            self.do_verify('stub', hash)
            result = self.do_genhash('stub', hash)
            self.assertIsInstance(result, str, 'genhash() failed to return native string: %r' % (result,))
        else:
            self.assertFalse(self.do_identify(hash), 'identify() incorrectly identified hash belonging to %s: %r' % (name, hash))
            self.assertRaises(ValueError, self.do_verify, 'stub', hash, __msg__='verify() failed to throw error for hash belonging to %s: %r' % (name, hash))
            self.assertRaises(ValueError, self.do_genhash, 'stub', hash, __msg__='genhash() failed to throw error for hash belonging to %s: %r' % (name, hash))