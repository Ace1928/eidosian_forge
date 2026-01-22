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
def test_run_call_order__failure_in_test_default_result(self):

    class Foo(Test.LoggingTestCase):

        def defaultTestResult(self):
            return LoggingResult(self.events)

        def test(self):
            super(Foo, self).test()
            self.fail('raised by Foo.test')
    expected = ['startTestRun', 'startTest', 'setUp', 'test', 'addFailure', 'tearDown', 'stopTest', 'stopTestRun']
    events = []
    Foo(events).run()
    self.assertEqual(events, expected)