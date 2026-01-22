import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_loops(self):
    for g in [self.loopless1(), self.loopless1_dead_nodes(), self.loopless2()]:
        self.assertEqual(len(g.loops()), 0)
    g = self.multiple_loops()
    self.assertEqual(sorted(g.loops()), [7, 20, 68])
    outer1 = g.loops()[7]
    inner1 = g.loops()[20]
    outer2 = g.loops()[68]
    self.assertEqual(outer1.header, 7)
    self.assertEqual(sorted(outer1.entries), [0])
    self.assertEqual(sorted(outer1.exits), [60])
    self.assertEqual(sorted(outer1.body), [7, 10, 13, 20, 23, 32, 44, 56, 57])
    self.assertEqual(inner1.header, 20)
    self.assertEqual(sorted(inner1.entries), [13])
    self.assertEqual(sorted(inner1.exits), [56])
    self.assertEqual(sorted(inner1.body), [20, 23, 32, 44])
    self.assertEqual(outer2.header, 68)
    self.assertEqual(sorted(outer2.entries), [61])
    self.assertEqual(sorted(outer2.exits), [80, 87])
    self.assertEqual(sorted(outer2.body), [68, 71])
    for node in [0, 60, 61, 80, 87, 88]:
        self.assertEqual(g.in_loops(node), [])
    for node in [7, 10, 13, 56, 57]:
        self.assertEqual(g.in_loops(node), [outer1])
    for node in [20, 23, 32, 44]:
        self.assertEqual(g.in_loops(node), [inner1, outer1])
    for node in [68, 71]:
        self.assertEqual(g.in_loops(node), [outer2])
    g = self.infinite_loop1()
    self.assertEqual(sorted(g.loops()), [13])
    loop = g.loops()[13]
    self.assertEqual(loop.header, 13)
    self.assertEqual(sorted(loop.entries), [10])
    self.assertEqual(sorted(loop.exits), [])
    self.assertEqual(sorted(loop.body), [13, 19, 26])
    for node in [0, 6, 10]:
        self.assertEqual(g.in_loops(node), [])
    for node in [13, 19, 26]:
        self.assertEqual(g.in_loops(node), [loop])
    g = self.infinite_loop2()
    self.assertEqual(sorted(g.loops()), [3])
    loop = g.loops()[3]
    self.assertEqual(loop.header, 3)
    self.assertEqual(sorted(loop.entries), [0])
    self.assertEqual(sorted(loop.exits), [])
    self.assertEqual(sorted(loop.body), [3, 9, 16])
    for node in [0]:
        self.assertEqual(g.in_loops(node), [])
    for node in [3, 9, 16]:
        self.assertEqual(g.in_loops(node), [loop])
    g = self.multiple_exits()
    self.assertEqual(sorted(g.loops()), [7])
    loop = g.loops()[7]
    self.assertEqual(loop.header, 7)
    self.assertEqual(sorted(loop.entries), [0])
    self.assertEqual(sorted(loop.exits), [19, 29, 36])
    self.assertEqual(sorted(loop.body), [7, 10, 23])
    for node in [0, 19, 29, 36]:
        self.assertEqual(g.in_loops(node), [])
    for node in [7, 10, 23]:
        self.assertEqual(g.in_loops(node), [loop])