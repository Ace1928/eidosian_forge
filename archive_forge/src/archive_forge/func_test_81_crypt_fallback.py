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
def test_81_crypt_fallback(self):
    """test per-call crypt() fallback"""
    mock_crypt = self._use_mock_crypt()
    mock_crypt.return_value = None
    if self.has_os_crypt_fallback:
        h1 = self.do_encrypt('stub')
        h2 = self.do_genhash('stub', h1)
        self.assertEqual(h2, h1)
        self.assertTrue(self.do_verify('stub', h1))
    else:
        from passlib.exc import InternalBackendError as err_type
        hash = self.get_sample_hash()[1]
        self.assertRaises(err_type, self.do_encrypt, 'stub')
        self.assertRaises(err_type, self.do_genhash, 'stub', hash)
        self.assertRaises(err_type, self.do_verify, 'stub', hash)