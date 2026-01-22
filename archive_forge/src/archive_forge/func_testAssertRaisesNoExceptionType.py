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
def testAssertRaisesNoExceptionType(self):
    with self.assertRaises(TypeError):
        self.assertRaises()
    with self.assertRaises(TypeError):
        self.assertRaises(1)
    with self.assertRaises(TypeError):
        self.assertRaises(object)
    with self.assertRaises(TypeError):
        self.assertRaises((ValueError, 1))
    with self.assertRaises(TypeError):
        self.assertRaises((ValueError, object))