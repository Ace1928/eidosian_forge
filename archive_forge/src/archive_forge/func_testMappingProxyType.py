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
def testMappingProxyType(self):
    structure = types.MappingProxyType({'a': 1, 'b': (2, 3)})
    expected = types.MappingProxyType({'a': 4, 'b': (5, 6)})
    self.assertEqual(tree.flatten(structure), [1, 2, 3])
    self.assertEqual(tree.unflatten_as(structure, [4, 5, 6]), expected)
    self.assertEqual(tree.map_structure(lambda v: v + 3, structure), expected)