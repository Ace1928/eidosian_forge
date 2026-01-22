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
def testAssertCountEqual(self):
    a = object()
    self.assertCountEqual([1, 2, 3], [3, 2, 1])
    self.assertCountEqual(['foo', 'bar', 'baz'], ['bar', 'baz', 'foo'])
    self.assertCountEqual([a, a, 2, 2, 3], (a, 2, 3, a, 2))
    self.assertCountEqual([1, '2', 'a', 'a'], ['a', '2', True, 'a'])
    self.assertRaises(self.failureException, self.assertCountEqual, [1, 2] + [3] * 100, [1] * 100 + [2, 3])
    self.assertRaises(self.failureException, self.assertCountEqual, [1, '2', 'a', 'a'], ['a', '2', True, 1])
    self.assertRaises(self.failureException, self.assertCountEqual, [10], [10, 11])
    self.assertRaises(self.failureException, self.assertCountEqual, [10, 11], [10])
    self.assertRaises(self.failureException, self.assertCountEqual, [10, 11, 10], [10, 11])
    self.assertCountEqual([[1, 2], [3, 4], 0], [False, [3, 4], [1, 2]])
    self.assertCountEqual(iter([1, 2, [], 3, 4]), iter([1, 2, [], 3, 4]))
    self.assertRaises(self.failureException, self.assertCountEqual, [], [divmod, 'x', 1, 5j, 2j, frozenset()])
    self.assertCountEqual([{'a': 1}, {'b': 2}], [{'b': 2}, {'a': 1}])
    self.assertCountEqual([1, 'x', divmod, []], [divmod, [], 'x', 1])
    self.assertRaises(self.failureException, self.assertCountEqual, [], [divmod, [], 'x', 1, 5j, 2j, set()])
    self.assertRaises(self.failureException, self.assertCountEqual, [[1]], [[2]])
    self.assertRaises(self.failureException, self.assertCountEqual, [1, 1, 2], [2, 1])
    self.assertRaises(self.failureException, self.assertCountEqual, [1, 1, '2', 'a', 'a'], ['2', '2', True, 'a'])
    self.assertRaises(self.failureException, self.assertCountEqual, [1, {'b': 2}, None, True], [{'b': 2}, True, None])
    a = [{2, 4}, {1, 2}]
    b = a[::-1]
    self.assertCountEqual(a, b)
    diffs = set(unittest.util._count_diff_all_purpose('aaabccd', 'abbbcce'))
    expected = {(3, 1, 'a'), (1, 3, 'b'), (1, 0, 'd'), (0, 1, 'e')}
    self.assertEqual(diffs, expected)
    diffs = unittest.util._count_diff_all_purpose([[]], [])
    self.assertEqual(diffs, [(1, 0, [])])
    diffs = set(unittest.util._count_diff_hashable('aaabccd', 'abbbcce'))
    expected = {(3, 1, 'a'), (1, 3, 'b'), (1, 0, 'd'), (0, 1, 'e')}
    self.assertEqual(diffs, expected)