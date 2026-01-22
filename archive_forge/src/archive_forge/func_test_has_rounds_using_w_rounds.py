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
def test_has_rounds_using_w_rounds(self):
    """
        HasRounds.using() -- rounds
        """
    handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
    orig_max_rounds = handler.max_rounds
    temp = subcls.using(rounds=medium + adj)
    self.assertEqual(temp.min_desired_rounds, medium + adj)
    self.assertEqual(temp.default_rounds, medium + adj)
    self.assertEqual(temp.max_desired_rounds, medium + adj)
    temp = subcls.using(rounds=medium + 1, min_rounds=small + adj, default_rounds=medium, max_rounds=large - adj)
    self.assertEqual(temp.min_desired_rounds, small + adj)
    self.assertEqual(temp.default_rounds, medium)
    self.assertEqual(temp.max_desired_rounds, large - adj)