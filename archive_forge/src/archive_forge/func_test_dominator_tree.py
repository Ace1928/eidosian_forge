import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_dominator_tree(self):

    def check(graph, expected):
        domtree = graph.dominator_tree()
        self.assertEqual(domtree, expected)
    check(self.loopless1(), {0: {12, 18, 21}, 12: set(), 18: set(), 21: set()})
    check(self.loopless2(), {12: set(), 18: set(), 21: {34, 42}, 34: set(), 42: set(), 99: {18, 12, 21}})
    check(self.loopless1_dead_nodes(), {0: {12, 18, 21}, 12: set(), 18: set(), 21: set()})
    check(self.multiple_loops(), {0: {7}, 7: {10, 60}, 60: {61}, 61: {68}, 68: {71, 87}, 87: {88}, 88: set(), 71: {80}, 80: set(), 10: {13}, 13: {20}, 20: {56, 23}, 23: {32, 44}, 44: set(), 32: set(), 56: {57}, 57: set()})
    check(self.multiple_exits(), {0: {7}, 7: {10, 36, 37}, 36: set(), 10: {19, 23}, 23: {29}, 29: set(), 37: set(), 19: set()})
    check(self.infinite_loop1(), {0: {10, 6}, 6: set(), 10: {13}, 13: {26, 19}, 19: set(), 26: set()})
    check(self.infinite_loop2(), {0: {3}, 3: {16, 9}, 9: set(), 16: set()})