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
def test_13_max_salt_size(self):
    """test hash() / genconfig() honors max_salt_size"""
    self.require_salt_info()
    handler = self.handler
    max_size = handler.max_salt_size
    salt_char = handler.salt_chars[0:1]
    if max_size is None or max_size > 1 << 20:
        s1 = salt_char * 1024
        c1 = self.do_stub_encrypt(salt=s1)
        c2 = self.do_stub_encrypt(salt=s1 + salt_char)
        self.assertNotEqual(c1, c2)
        self.do_stub_encrypt(salt_size=1024)
    else:
        s1 = salt_char * max_size
        c1 = self.do_stub_encrypt(salt=s1)
        self.do_stub_encrypt(salt_size=max_size)
        s2 = s1 + salt_char
        self.assertRaises(ValueError, self.do_stub_encrypt, salt=s2)
        self.assertRaises(ValueError, self.do_stub_encrypt, salt_size=max_size + 1)
        if has_relaxed_setting(handler):
            with warnings.catch_warnings(record=True):
                c2 = self.do_stub_encrypt(salt=s2, relaxed=True)
            self.assertEqual(c2, c1)
        if handler.min_salt_size < max_size:
            c3 = self.do_stub_encrypt(salt=s1[:-1])
            self.assertNotEqual(c3, c1)