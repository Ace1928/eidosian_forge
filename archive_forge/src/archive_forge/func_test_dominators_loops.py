import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_dominators_loops(self):
    g = self.multiple_exits()
    doms = g.dominators()
    self.check_dominators(doms, {0: [0], 7: [0, 7], 10: [0, 7, 10], 19: [0, 7, 10, 19], 23: [0, 7, 10, 23], 29: [0, 7, 10, 23, 29], 36: [0, 7, 36], 37: [0, 7, 37]})
    g = self.multiple_loops()
    doms = g.dominators()
    self.check_dominators(doms, {0: [0], 7: [0, 7], 10: [0, 10, 7], 13: [0, 10, 13, 7], 20: [0, 10, 20, 13, 7], 23: [0, 20, 23, 7, 10, 13], 32: [32, 0, 20, 23, 7, 10, 13], 44: [0, 20, 23, 7, 10, 44, 13], 56: [0, 20, 7, 56, 10, 13], 57: [0, 20, 7, 56, 57, 10, 13], 60: [0, 60, 7], 61: [0, 60, 61, 7], 68: [0, 68, 60, 61, 7], 71: [0, 68, 71, 7, 60, 61], 80: [80, 0, 68, 71, 7, 60, 61], 87: [0, 68, 87, 7, 60, 61], 88: [0, 68, 87, 88, 7, 60, 61]})
    g = self.infinite_loop1()
    doms = g.dominators()
    self.check_dominators(doms, {0: [0], 6: [0, 6], 10: [0, 10], 13: [0, 10, 13], 19: [0, 10, 19, 13], 26: [0, 10, 13, 26]})