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
def testFlattenUpTo(self):
    input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
    shallow_tree = [[True, True], [False, True]]
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [[2, 2], [3, 3], [4, 9], [5, 5]])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])
    input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
    shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree, input_tree)
    input_tree_flattened = tree.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [('a', 1), ('b', 2), ('c', 3), ('d', 4)])
    self.assertEqual(input_tree_flattened, ['a', 1, 'b', 2, 'c', 3, 'd', 4])
    input_tree = {'a': 1, 'b': {'c': 2}, 'd': [3, (4, 5)]}
    shallow_tree = {'a': 0, 'b': 0, 'd': [0, 0]}
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree, input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [1, {'c': 2}, 3, (4, 5)])
    ab_tuple = collections.namedtuple('ab_tuple', 'a, b')
    input_tree = ab_tuple(a=[0, 1], b=2)
    shallow_tree = ab_tuple(a=0, b=1)
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree, input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [[0, 1], 2])

    @attr.s
    class ABAttr(object):
        a = attr.ib()
        b = attr.ib()
    input_tree = ABAttr(a=[0, 1], b=2)
    shallow_tree = ABAttr(a=0, b=1)
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree, input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [[0, 1], 2])
    input_tree = collections.OrderedDict([('a', ab_tuple(a=[0, {'b': 1}], b=2)), ('c', {'d': 3, 'e': collections.OrderedDict([('f', 4)])})])
    shallow_tree = input_tree
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree, input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [0, 1, 2, 3, 4])
    shallow_tree = collections.OrderedDict([('a', 0), ('c', {'d': 3, 'e': 1})])
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree, input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [ab_tuple(a=[0, {'b': 1}], b=2), 3, collections.OrderedDict([('f', 4)])])
    shallow_tree = collections.OrderedDict([('a', 0), ('c', 0)])
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree, input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [ab_tuple(a=[0, {'b': 1}], b=2), {'d': 3, 'e': collections.OrderedDict([('f', 4)])}])
    input_tree = ['input_tree']
    shallow_tree = 'shallow_tree'
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])
    input_tree = ['input_tree_0', 'input_tree_1']
    shallow_tree = 'shallow_tree'
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])
    input_tree = [0]
    shallow_tree = 9
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])
    input_tree = [0, 1]
    shallow_tree = 9
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])
    input_tree = 'input_tree'
    shallow_tree = 'shallow_tree'
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])
    input_tree = 0
    shallow_tree = 0
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])
    input_tree = 'input_tree'
    shallow_tree = ['shallow_tree']
    with self.assertRaisesWithLiteralMatch(TypeError, tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree))):
        flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)
    input_tree = 'input_tree'
    shallow_tree = ['shallow_tree_9', 'shallow_tree_8']
    with self.assertRaisesWithLiteralMatch(TypeError, tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree))):
        flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)
    input_tree = 0
    shallow_tree = [9]
    with self.assertRaisesWithLiteralMatch(TypeError, tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree))):
        flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)
    input_tree = 0
    shallow_tree = [9, 8]
    with self.assertRaisesWithLiteralMatch(TypeError, tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree))):
        flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)