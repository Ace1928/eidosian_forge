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
def testEqualityBytesWarning(self):
    if sys.flags.bytes_warning:

        def bytes_warning():
            return self.assertWarnsRegex(BytesWarning, 'Comparison between bytes and string')
    else:

        def bytes_warning():
            return contextlib.ExitStack()
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertEqual('a', b'a')
    with bytes_warning():
        self.assertNotEqual('a', b'a')
    a = [0, 'a']
    b = [0, b'a']
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertListEqual(a, b)
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertTupleEqual(tuple(a), tuple(b))
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertSequenceEqual(a, tuple(b))
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertSequenceEqual(tuple(a), b)
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertSequenceEqual('a', b'a')
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertSetEqual(set(a), set(b))
    with self.assertRaises(self.failureException):
        self.assertListEqual(a, tuple(b))
    with self.assertRaises(self.failureException):
        self.assertTupleEqual(tuple(a), b)
    a = [0, b'a']
    b = [0]
    with self.assertRaises(self.failureException):
        self.assertListEqual(a, b)
    with self.assertRaises(self.failureException):
        self.assertTupleEqual(tuple(a), tuple(b))
    with self.assertRaises(self.failureException):
        self.assertSequenceEqual(a, tuple(b))
    with self.assertRaises(self.failureException):
        self.assertSequenceEqual(tuple(a), b)
    with self.assertRaises(self.failureException):
        self.assertSetEqual(set(a), set(b))
    a = [0]
    b = [0, b'a']
    with self.assertRaises(self.failureException):
        self.assertListEqual(a, b)
    with self.assertRaises(self.failureException):
        self.assertTupleEqual(tuple(a), tuple(b))
    with self.assertRaises(self.failureException):
        self.assertSequenceEqual(a, tuple(b))
    with self.assertRaises(self.failureException):
        self.assertSequenceEqual(tuple(a), b)
    with self.assertRaises(self.failureException):
        self.assertSetEqual(set(a), set(b))
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertDictEqual({'a': 0}, {b'a': 0})
    with self.assertRaises(self.failureException):
        self.assertDictEqual({}, {b'a': 0})
    with self.assertRaises(self.failureException):
        self.assertDictEqual({b'a': 0}, {})
    with self.assertRaises(self.failureException):
        self.assertCountEqual([b'a', b'a'], [b'a', b'a', b'a'])
    with bytes_warning():
        self.assertCountEqual(['a', b'a'], ['a', b'a'])
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertCountEqual(['a', 'a'], [b'a', b'a'])
    with bytes_warning(), self.assertRaises(self.failureException):
        self.assertCountEqual(['a', 'a', []], [b'a', b'a', []])