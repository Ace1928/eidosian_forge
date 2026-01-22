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
def testAssertRaisesContext(self):

    class ExceptionMock(Exception):
        pass

    def Stub():
        raise ExceptionMock('We expect')
    with self.assertRaises(ExceptionMock):
        Stub()
    with self.assertRaises((ValueError, ExceptionMock)) as cm:
        Stub()
    self.assertIsInstance(cm.exception, ExceptionMock)
    self.assertEqual(cm.exception.args[0], 'We expect')
    with self.assertRaises(ValueError):
        int('19', base=8)
    with self.assertRaises(self.failureException):
        with self.assertRaises(ExceptionMock):
            pass
    with self.assertRaisesRegex(self.failureException, 'foobar'):
        with self.assertRaises(ExceptionMock, msg='foobar'):
            pass
    with self.assertRaisesRegex(TypeError, 'foobar'):
        with self.assertRaises(ExceptionMock, foobar=42):
            pass
    with self.assertRaises(ExceptionMock):
        self.assertRaises(ValueError, Stub)