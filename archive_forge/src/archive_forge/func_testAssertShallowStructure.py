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
def testAssertShallowStructure(self):
    inp_ab = ['a', 'b']
    inp_abc = ['a', 'b', 'c']
    with self.assertRaisesRegex(ValueError, tree._STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(input_length=len(inp_ab), shallow_length=len(inp_abc))):
        tree._assert_shallow_structure(inp_abc, inp_ab)
    inp_ab1 = [(1, 1), (2, 2)]
    inp_ab2 = [[1, 1], [2, 2]]
    with self.assertRaisesWithLiteralMatch(TypeError, tree._STRUCTURES_HAVE_MISMATCHING_TYPES.format(shallow_type=type(inp_ab2[0]), input_type=type(inp_ab1[0]))):
        tree._assert_shallow_structure(shallow_tree=inp_ab2, input_tree=inp_ab1)
    tree._assert_shallow_structure(inp_ab2, inp_ab1, check_types=False)
    inp_ab1 = {'a': (1, 1), 'b': {'c': (2, 2)}}
    inp_ab2 = {'a': (1, 1), 'b': {'d': (2, 2)}}
    with self.assertRaisesWithLiteralMatch(ValueError, tree._SHALLOW_TREE_HAS_INVALID_KEYS.format(['d'])):
        tree._assert_shallow_structure(inp_ab2, inp_ab1)
    inp_ab = collections.OrderedDict([('a', 1), ('b', (2, 3))])
    inp_ba = collections.OrderedDict([('b', (2, 3)), ('a', 1)])
    tree._assert_shallow_structure(inp_ab, inp_ba)
    tree._assert_shallow_structure({0: 'foo'}, ['foo'], check_types=False)