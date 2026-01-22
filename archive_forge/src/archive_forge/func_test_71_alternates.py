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
def test_71_alternates(self):
    """test known alternate hashes"""
    if not self.known_alternate_hashes:
        raise self.skipTest('no alternate hashes provided')
    for alt, secret, hash in self.known_alternate_hashes:
        self.assertTrue(self.do_identify(hash), 'identify() failed to identify alternate hash: %r' % (hash,))
        self.check_verify(secret, alt, 'verify() of known alternate hash failed: secret=%r, hash=%r' % (secret, alt))
        result = self.do_genhash(secret, alt)
        self.assertIsInstance(result, str, 'genhash() failed to return native string: %r' % (result,))
        if self.handler.is_disabled and self.disabled_contains_salt:
            continue
        self.assertEqual(result, hash, 'genhash() failed to normalize known alternate hash: secret=%r, alt=%r, hash=%r: result=%r' % (secret, alt, hash, result))