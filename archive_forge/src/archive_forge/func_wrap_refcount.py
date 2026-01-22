from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
def wrap_refcount(method):
    if getattr(method, 'ignore_leakcheck', False) or SKIP_LEAKCHECKS:
        return method

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, 'ignore_leakcheck', False):
            raise unittest.SkipTest('This class ignored during leakchecks')
        if ONLY_FAILING_LEAKCHECKS and (not getattr(method, 'fails_leakcheck', False)):
            raise unittest.SkipTest('Only running tests that fail leakchecks.')
        return _RefCountChecker(self, method)(args, kwargs)
    return wrapper