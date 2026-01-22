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
def testMapStructureUpTo(self):
    ab_tuple = collections.namedtuple('ab_tuple', 'a, b')
    op_tuple = collections.namedtuple('op_tuple', 'add, mul')
    inp_val = ab_tuple(a=2, b=3)
    inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
    out = tree.map_structure_up_to(inp_val, lambda val, ops: (val + ops.add) * ops.mul, inp_val, inp_ops, check_types=False)
    self.assertEqual(out.a, 6)
    self.assertEqual(out.b, 15)
    data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
    name_list = ['evens', ['odds', 'primes']]
    out = tree.map_structure_up_to(name_list, lambda name, sec: 'first_{}_{}'.format(len(sec), name), name_list, data_list)
    self.assertEqual(out, ['first_4_evens', ['first_5_odds', 'first_3_primes']])