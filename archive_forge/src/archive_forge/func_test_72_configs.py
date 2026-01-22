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
def test_72_configs(self):
    """test known config strings"""
    if not self.handler.setting_kwds:
        self.assertFalse(self.known_correct_configs, 'handler should not have config strings')
        raise self.skipTest('hash has no settings')
    if not self.known_correct_configs:
        raise self.skipTest('no config strings provided')
    if self.filter_config_warnings:
        warnings.filterwarnings('ignore', category=PasslibHashWarning)
    for config, secret, hash in self.known_correct_configs:
        self.assertTrue(self.do_identify(config), 'identify() failed to identify known config string: %r' % (config,))
        self.assertRaises(ValueError, self.do_verify, secret, config, __msg__='verify() failed to reject config string: %r' % (config,))
        result = self.do_genhash(secret, config)
        self.assertIsInstance(result, str, 'genhash() failed to return native string: %r' % (result,))
        self.assertEqual(result, hash, 'genhash() failed to reproduce known hash from config: secret=%r, config=%r, hash=%r: result=%r' % (secret, config, hash, result))