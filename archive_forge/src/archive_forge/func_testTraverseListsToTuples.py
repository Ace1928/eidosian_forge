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
def testTraverseListsToTuples(self):
    structure = [(1, 2), [3], {'a': [4]}]
    self.assertEqual(((1, 2), (3,), {'a': (4,)}), tree.traverse(lambda x: tuple(x) if isinstance(x, list) else x, structure, top_down=False))