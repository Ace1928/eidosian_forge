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
def test_01_required_attributes(self):
    """validate required attributes"""
    handler = self.handler

    def ga(name):
        return getattr(handler, name, None)
    name = ga('name')
    self.assertTrue(name, 'name not defined:')
    self.assertIsInstance(name, str, 'name must be native str')
    self.assertTrue(name.lower() == name, 'name not lower-case:')
    self.assertTrue(re.match('^[a-z0-9_]+$', name), 'name must be alphanum + underscore: %r' % (name,))
    settings = ga('setting_kwds')
    self.assertTrue(settings is not None, 'setting_kwds must be defined:')
    self.assertIsInstance(settings, tuple, 'setting_kwds must be a tuple:')
    context = ga('context_kwds')
    self.assertTrue(context is not None, 'context_kwds must be defined:')
    self.assertIsInstance(context, tuple, 'context_kwds must be a tuple:')