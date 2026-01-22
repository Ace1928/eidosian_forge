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
def test_init__test_name__valid(self):

    class Test(unittest.TestCase):

        def runTest(self):
            raise MyException()

        def test(self):
            pass
    self.assertEqual(Test('test').id()[-10:], '.Test.test')