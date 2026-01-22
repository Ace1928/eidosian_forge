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
def test_02_config_workflow(self):
    """test basic config-string workflow

        this tests that genconfig() returns the expected types,
        and that identify() and genhash() handle the result correctly.
        """
    config = self.do_genconfig()
    self.check_returned_native_str(config, 'genconfig')
    result = self.do_genhash('stub', config)
    self.check_returned_native_str(result, 'genhash')
    self.do_verify('', config)
    self.assertTrue(self.do_identify(config), 'identify() failed to identify genconfig() output: %r' % (config,))