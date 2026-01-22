import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
def testAssertWarnsCallable(self):

    def _runtime_warn():
        warnings.warn('foo', RuntimeWarning)
    self.assertWarns(RuntimeWarning, _runtime_warn)
    self.assertWarns(RuntimeWarning, _runtime_warn)
    self.assertWarns((DeprecationWarning, RuntimeWarning), _runtime_warn)
    self.assertWarns(RuntimeWarning, warnings.warn, 'foo', category=RuntimeWarning)
    with self.assertRaises(self.failureException):
        self.assertWarns(RuntimeWarning, lambda: 0)
    with self.assertRaises(TypeError):
        self.assertWarns(RuntimeWarning, None)
    with warnings.catch_warnings():
        warnings.simplefilter('default', RuntimeWarning)
        with self.assertRaises(self.failureException):
            self.assertWarns(DeprecationWarning, _runtime_warn)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with self.assertRaises(RuntimeWarning):
            self.assertWarns(DeprecationWarning, _runtime_warn)