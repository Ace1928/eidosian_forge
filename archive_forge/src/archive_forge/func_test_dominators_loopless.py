import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_dominators_loopless(self):

    def eq_(d, l):
        self.assertEqual(sorted(doms[d]), l)
    for g in [self.loopless1(), self.loopless1_dead_nodes()]:
        doms = g.dominators()
        eq_(0, [0])
        eq_(12, [0, 12])
        eq_(18, [0, 18])
        eq_(21, [0, 21])
    g = self.loopless2()
    doms = g.dominators()
    eq_(99, [99])
    eq_(12, [12, 99])
    eq_(18, [18, 99])
    eq_(21, [21, 99])
    eq_(34, [21, 34, 99])
    eq_(42, [21, 42, 99])