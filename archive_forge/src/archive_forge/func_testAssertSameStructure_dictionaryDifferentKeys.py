import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest
from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt
def testAssertSameStructure_dictionaryDifferentKeys(self):
    with self.assertRaisesRegex(ValueError, "don't have the same set of keys"):
        tree.assert_same_structure({'a': 1}, {'b': 1})