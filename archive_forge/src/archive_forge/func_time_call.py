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
def time_call(func, setup=None, maxtime=1, bestof=10):
    """
    timeit() wrapper which tries to get as accurate a measurement as possible w/in maxtime seconds.

    :returns:
        ``(avg_seconds_per_call, log10_number_of_repetitions)``
    """
    from timeit import Timer
    from math import log
    timer = Timer(func, setup=setup or '')
    number = 1
    end = tick() + maxtime
    while True:
        delta = min(timer.repeat(bestof, number))
        if tick() >= end:
            return (delta / number, int(log(number, 10)))
        number *= 10