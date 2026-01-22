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
def test_has_many_idents_using(self):
    """HasManyIdents.using() -- 'default_ident' and 'ident' keywords"""
    self.require_many_idents()
    handler = self.handler
    orig_ident = handler.default_ident
    for alt_ident in handler.ident_values:
        if alt_ident != orig_ident:
            break
    else:
        raise AssertionError('expected to find alternate ident: default=%r values=%r' % (orig_ident, handler.ident_values))

    def effective_ident(cls):
        cls = unwrap_handler(cls)
        return cls(use_defaults=True).ident
    subcls = handler.using()
    self.assertEqual(subcls.default_ident, orig_ident)
    subcls = handler.using(default_ident=alt_ident)
    self.assertEqual(subcls.default_ident, alt_ident)
    self.assertEqual(handler.default_ident, orig_ident)
    self.assertEqual(effective_ident(subcls), alt_ident)
    self.assertEqual(effective_ident(handler), orig_ident)
    self.assertRaises(ValueError, handler.using, default_ident='xXx')
    subcls = handler.using(ident=alt_ident)
    self.assertEqual(subcls.default_ident, alt_ident)
    self.assertEqual(handler.default_ident, orig_ident)
    self.assertRaises(TypeError, handler.using, default_ident=alt_ident, ident=alt_ident)
    if handler.ident_aliases:
        for alias, ident in handler.ident_aliases.items():
            subcls = handler.using(ident=alias)
            self.assertEqual(subcls.default_ident, ident, msg='alias %r:' % alias)