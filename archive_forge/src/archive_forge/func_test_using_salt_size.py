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
def test_using_salt_size(self):
    """Handler.using() -- default_salt_size"""
    self.require_salt_info()
    handler = self.handler
    mn = handler.min_salt_size
    mx = handler.max_salt_size
    df = handler.default_salt_size
    self.assertRaises(ValueError, handler.using, default_salt_size=-1)
    with self.assertWarningList([PasslibHashWarning]):
        temp = handler.using(default_salt_size=-1, relaxed=True)
    self.assertEqual(temp.default_salt_size, mn)
    if mx:
        self.assertRaises(ValueError, handler.using, default_salt_size=mx + 1)
        with self.assertWarningList([PasslibHashWarning]):
            temp = handler.using(default_salt_size=mx + 1, relaxed=True)
        self.assertEqual(temp.default_salt_size, mx)
    if mn != mx:
        temp = handler.using(default_salt_size=mn + 1)
        self.assertEqual(temp.default_salt_size, mn + 1)
        self.assertEqual(handler.default_salt_size, df)
        temp = handler.using(default_salt_size=mn + 2)
        self.assertEqual(temp.default_salt_size, mn + 2)
        self.assertEqual(handler.default_salt_size, df)
    if mn == mx:
        ref = mn
    else:
        ref = mn + 1
    temp = handler.using(default_salt_size=str(ref))
    self.assertEqual(temp.default_salt_size, ref)
    self.assertRaises(ValueError, handler.using, default_salt_size=str(ref) + 'xxx')
    temp = handler.using(salt_size=ref)
    self.assertEqual(temp.default_salt_size, ref)