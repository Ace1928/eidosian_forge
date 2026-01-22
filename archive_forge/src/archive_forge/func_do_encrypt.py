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
def do_encrypt(self, secret, use_encrypt=False, handler=None, context=None, **settings):
    """call handler's hash() method with specified options"""
    self.populate_settings(settings)
    if context is None:
        context = {}
    secret = self.populate_context(secret, context)
    if use_encrypt:
        warnings = []
        if settings:
            context.update(**settings)
            warnings.append('passing settings to.*is deprecated')
        with self.assertWarningList(warnings):
            return (handler or self.handler).encrypt(secret, **context)
    else:
        return (handler or self.handler).using(**settings).hash(secret, **context)