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
def checkAssertLogsPerLevel(self, level):
    with self.assertNoStderr():
        with self.assertLogs(level=level) as cm:
            log_foo.warning('1')
            log_foobar.error('2')
            log_quux.critical('3')
        self.assertEqual(cm.output, ['ERROR:foo.bar:2', 'CRITICAL:quux:3'])
        self.assertLogRecords(cm.records, [{'name': 'foo.bar'}, {'name': 'quux'}])