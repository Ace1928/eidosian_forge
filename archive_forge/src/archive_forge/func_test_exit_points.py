import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_exit_points(self):
    g = self.loopless1()
    self.assertEqual(sorted(g.exit_points()), [21])
    g = self.loopless1_dead_nodes()
    self.assertEqual(sorted(g.exit_points()), [21])
    g = self.loopless2()
    self.assertEqual(sorted(g.exit_points()), [34, 42])
    g = self.multiple_loops()
    self.assertEqual(sorted(g.exit_points()), [80, 88])
    g = self.infinite_loop1()
    self.assertEqual(sorted(g.exit_points()), [6])
    g = self.infinite_loop2()
    self.assertEqual(sorted(g.exit_points()), [])
    g = self.multiple_exits()
    self.assertEqual(sorted(g.exit_points()), [19, 37])