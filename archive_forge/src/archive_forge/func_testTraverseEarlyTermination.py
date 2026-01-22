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
def testTraverseEarlyTermination(self):
    structure = [(1, [2]), [3, (4, 5, 6)]]
    visited = []

    def visit(x):
        visited.append(x)
        return 'X' if isinstance(x, tuple) and len(x) > 2 else None
    output = tree.traverse(visit, structure)
    self.assertEqual([(1, [2]), [3, 'X']], output)
    self.assertEqual([[(1, [2]), [3, (4, 5, 6)]], (1, [2]), 1, [2], 2, [3, (4, 5, 6)], 3, (4, 5, 6)], visited)