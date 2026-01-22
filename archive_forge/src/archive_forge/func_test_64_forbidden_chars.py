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
def test_64_forbidden_chars(self):
    """test forbidden characters not allowed in password"""
    chars = self.forbidden_characters
    if not chars:
        raise self.skipTest('none listed')
    base = u('stub')
    if isinstance(chars, bytes):
        from passlib.utils.compat import iter_byte_chars
        chars = iter_byte_chars(chars)
        base = base.encode('ascii')
    for c in chars:
        self.assertRaises(ValueError, self.do_encrypt, base + c + base)