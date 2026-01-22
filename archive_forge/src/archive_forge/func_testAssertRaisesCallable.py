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
def testAssertRaisesCallable(self):

    class ExceptionMock(Exception):
        pass

    def Stub():
        raise ExceptionMock('We expect')
    self.assertRaises(ExceptionMock, Stub)
    self.assertRaises((ValueError, ExceptionMock), Stub)
    self.assertRaises(ValueError, int, '19', base=8)
    with self.assertRaises(self.failureException):
        self.assertRaises(ExceptionMock, lambda: 0)
    with self.assertRaises(TypeError):
        self.assertRaises(ExceptionMock, None)
    with self.assertRaises(ExceptionMock):
        self.assertRaises(ValueError, Stub)