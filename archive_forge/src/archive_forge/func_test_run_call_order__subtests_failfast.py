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
def test_run_call_order__subtests_failfast(self):
    events = []
    result = LoggingResult(events)
    result.failfast = True

    class Foo(Test.LoggingTestCase):

        def test(self):
            super(Foo, self).test()
            with self.subTest(i=1):
                self.fail('failure')
            with self.subTest(i=2):
                self.fail('failure')
            self.fail('failure')
    expected = ['startTest', 'setUp', 'test', 'addSubTestFailure', 'tearDown', 'stopTest']
    Foo(events).run(result)
    self.assertEqual(events, expected)