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
def test_call__invoking_an_instance_delegates_to_run(self):
    resultIn = unittest.TestResult()
    resultOut = unittest.TestResult()

    class Foo(unittest.TestCase):

        def test(self):
            pass

        def run(self, result):
            self.assertIs(result, resultIn)
            return resultOut
    retval = Foo('test')(resultIn)
    self.assertIs(retval, resultOut)