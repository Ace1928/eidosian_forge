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
def testMapStructureAcrossSubtreesNoneValues(self):
    shallow = [1, [None]]
    deep1 = [1, [2, 3]]
    deep2 = [2, [3, 4]]
    summed = tree.map_structure_up_to(shallow, lambda *args: sum(args), deep1, deep2)
    expected = [3, [5]]
    self.assertEqual(summed, expected)