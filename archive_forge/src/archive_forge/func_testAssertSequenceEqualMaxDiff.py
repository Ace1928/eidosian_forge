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
def testAssertSequenceEqualMaxDiff(self):
    self.assertEqual(self.maxDiff, 80 * 8)
    seq1 = 'a' + 'x' * 80 ** 2
    seq2 = 'b' + 'x' * 80 ** 2
    diff = '\n'.join(difflib.ndiff(pprint.pformat(seq1).splitlines(), pprint.pformat(seq2).splitlines()))
    omitted = unittest.case.DIFF_OMITTED % (len(diff) + 1,)
    self.maxDiff = len(diff) // 2
    try:
        self.assertSequenceEqual(seq1, seq2)
    except self.failureException as e:
        msg = e.args[0]
    else:
        self.fail('assertSequenceEqual did not fail.')
    self.assertLess(len(msg), len(diff))
    self.assertIn(omitted, msg)
    self.maxDiff = len(diff) * 2
    try:
        self.assertSequenceEqual(seq1, seq2)
    except self.failureException as e:
        msg = e.args[0]
    else:
        self.fail('assertSequenceEqual did not fail.')
    self.assertGreater(len(msg), len(diff))
    self.assertNotIn(omitted, msg)
    self.maxDiff = None
    try:
        self.assertSequenceEqual(seq1, seq2)
    except self.failureException as e:
        msg = e.args[0]
    else:
        self.fail('assertSequenceEqual did not fail.')
    self.assertGreater(len(msg), len(diff))
    self.assertNotIn(omitted, msg)