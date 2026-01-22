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
def test_05_backends(self):
    """test multi-backend support"""
    handler = self.handler
    if not hasattr(handler, 'set_backend'):
        raise self.skipTest('handler only has one backend')
    self.addCleanup(handler.set_backend, handler.get_backend())
    for backend in handler.backends:
        self.assertIsInstance(backend, str)
        self.assertNotIn(backend, RESERVED_BACKEND_NAMES, 'invalid backend name: %r' % (backend,))
        ret = handler.has_backend(backend)
        if ret is True:
            handler.set_backend(backend)
            self.assertEqual(handler.get_backend(), backend)
        elif ret is False:
            self.assertRaises(MissingBackendError, handler.set_backend, backend)
        else:
            raise TypeError('has_backend(%r) returned invalid value: %r' % (backend, ret))