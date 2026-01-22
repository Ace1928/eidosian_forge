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
def getRandom(self, name='default', seed=None):
    """
        Return a :class:`random.Random` object for current test method to use.
        Within an instance, multiple calls with the same name will return
        the same object.

        When first created, each RNG will be seeded with value derived from
        a global seed, the test class module & name, the current test method name,
        and the **name** parameter.

        The global seed taken from the $RANDOM_TEST_SEED env var,
        the $PYTHONHASHSEED env var, or a randomly generated the
        first time this method is called. In all cases, the value
        is logged for reproducibility.

        :param name:
            name to uniquely identify separate RNGs w/in a test
            (e.g. for threaded tests).

        :param seed:
            override global seed when initialzing rng.

        :rtype: random.Random
        """
    cache = self._random_cache
    if cache and name in cache:
        return cache[name]
    with self._random_global_lock:
        cache = self._random_cache
        if cache and name in cache:
            return cache[name]
        elif not cache:
            cache = self._random_cache = {}
        global_seed = seed or TestCase._random_global_seed
        if global_seed is None:
            global_seed = TestCase._random_global_seed = int(os.environ.get('RANDOM_TEST_SEED') or os.environ.get('PYTHONHASHSEED') or sys_rng.getrandbits(32))
            log.info('using RANDOM_TEST_SEED=%d', global_seed)
        cls = type(self)
        source = '\n'.join([str(global_seed), cls.__module__, cls.__name__, self._testMethodName, name])
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        seed = int(digest[:16], 16)
        value = cache[name] = random.Random(seed)
        return value