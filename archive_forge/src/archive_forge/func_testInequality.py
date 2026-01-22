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
def testInequality(self):
    self.assertGreater(2, 1)
    self.assertGreaterEqual(2, 1)
    self.assertGreaterEqual(1, 1)
    self.assertLess(1, 2)
    self.assertLessEqual(1, 2)
    self.assertLessEqual(1, 1)
    self.assertRaises(self.failureException, self.assertGreater, 1, 2)
    self.assertRaises(self.failureException, self.assertGreater, 1, 1)
    self.assertRaises(self.failureException, self.assertGreaterEqual, 1, 2)
    self.assertRaises(self.failureException, self.assertLess, 2, 1)
    self.assertRaises(self.failureException, self.assertLess, 1, 1)
    self.assertRaises(self.failureException, self.assertLessEqual, 2, 1)
    self.assertGreater(1.1, 1.0)
    self.assertGreaterEqual(1.1, 1.0)
    self.assertGreaterEqual(1.0, 1.0)
    self.assertLess(1.0, 1.1)
    self.assertLessEqual(1.0, 1.1)
    self.assertLessEqual(1.0, 1.0)
    self.assertRaises(self.failureException, self.assertGreater, 1.0, 1.1)
    self.assertRaises(self.failureException, self.assertGreater, 1.0, 1.0)
    self.assertRaises(self.failureException, self.assertGreaterEqual, 1.0, 1.1)
    self.assertRaises(self.failureException, self.assertLess, 1.1, 1.0)
    self.assertRaises(self.failureException, self.assertLess, 1.0, 1.0)
    self.assertRaises(self.failureException, self.assertLessEqual, 1.1, 1.0)
    self.assertGreater('bug', 'ant')
    self.assertGreaterEqual('bug', 'ant')
    self.assertGreaterEqual('ant', 'ant')
    self.assertLess('ant', 'bug')
    self.assertLessEqual('ant', 'bug')
    self.assertLessEqual('ant', 'ant')
    self.assertRaises(self.failureException, self.assertGreater, 'ant', 'bug')
    self.assertRaises(self.failureException, self.assertGreater, 'ant', 'ant')
    self.assertRaises(self.failureException, self.assertGreaterEqual, 'ant', 'bug')
    self.assertRaises(self.failureException, self.assertLess, 'bug', 'ant')
    self.assertRaises(self.failureException, self.assertLess, 'ant', 'ant')
    self.assertRaises(self.failureException, self.assertLessEqual, 'bug', 'ant')
    self.assertGreater(b'bug', b'ant')
    self.assertGreaterEqual(b'bug', b'ant')
    self.assertGreaterEqual(b'ant', b'ant')
    self.assertLess(b'ant', b'bug')
    self.assertLessEqual(b'ant', b'bug')
    self.assertLessEqual(b'ant', b'ant')
    self.assertRaises(self.failureException, self.assertGreater, b'ant', b'bug')
    self.assertRaises(self.failureException, self.assertGreater, b'ant', b'ant')
    self.assertRaises(self.failureException, self.assertGreaterEqual, b'ant', b'bug')
    self.assertRaises(self.failureException, self.assertLess, b'bug', b'ant')
    self.assertRaises(self.failureException, self.assertLess, b'ant', b'ant')
    self.assertRaises(self.failureException, self.assertLessEqual, b'bug', b'ant')