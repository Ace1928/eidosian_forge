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
def testAssertLogsTwoMatchingMessages(self):
    with self.assertNoStderr():
        with self.assertLogs() as cm:
            log_foo.info('1')
            log_foobar.debug('2')
            log_quux.warning('3')
        self.assertEqual(cm.output, ['INFO:foo:1', 'WARNING:quux:3'])
        self.assertLogRecords(cm.records, [{'name': 'foo'}, {'name': 'quux'}])