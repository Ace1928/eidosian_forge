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
def test_subtests_debug(self):
    events = []

    class Foo(unittest.TestCase):

        def test_a(self):
            events.append('test case')
            with self.subTest():
                events.append('subtest 1')
    Foo('test_a').debug()
    self.assertEqual(events, ['test case', 'subtest 1'])