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
def test_secret_wo_truncate_size(self):
    """
        test no password size limits enforced (if truncate_size=None)
        """
    hasher = self.handler
    if hasher.truncate_size is not None:
        self.assertGreaterEqual(hasher.truncate_size, 1)
        raise self.skipTest('truncate_size is set')
    secret = 'too many secrets' * 16
    alt = 'x'
    hash = self.do_encrypt(secret)
    verify_success = not hasher.is_disabled
    self.assertEqual(self.do_verify(secret, hash), verify_success, msg='verify rejected correct secret')
    alt_secret = secret[:-1] + alt
    self.assertFalse(self.do_verify(alt_secret, hash), 'full password not used in digest')