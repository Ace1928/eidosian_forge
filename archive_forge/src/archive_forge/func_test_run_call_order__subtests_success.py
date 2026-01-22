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
def test_run_call_order__subtests_success(self):
    events = []
    result = LoggingResult(events)
    expected = ['startTest', 'setUp', 'test'] + 6 * ['addSubTestSuccess'] + ['tearDown', 'addSuccess', 'stopTest']
    self._check_call_order__subtests_success(result, events, expected)