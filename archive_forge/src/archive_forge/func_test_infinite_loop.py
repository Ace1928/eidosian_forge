import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_infinite_loop(self):

    def foo():
        SET_BLOCK_A
        while True:
            if SET_BLOCK_B:
                SET_BLOCK_C
                return
            SET_BLOCK_D
        SET_BLOCK_E
    cfa, blkpts = self.get_cfa_and_namedblocks(foo)
    idoms = cfa.graph.immediate_dominators()
    if PYVERSION >= (3, 10):
        self.assertNotIn('E', blkpts)
    else:
        self.assertNotIn(blkpts['E'], idoms)
    self.assertEqual(blkpts['B'], idoms[blkpts['C']])
    self.assertEqual(blkpts['B'], idoms[blkpts['D']])
    domfront = cfa.graph.dominance_frontier()
    if PYVERSION < (3, 10):
        self.assertNotIn(blkpts['E'], domfront)
    self.assertFalse(domfront[blkpts['A']])
    self.assertFalse(domfront[blkpts['C']])
    self.assertEqual({blkpts['B']}, domfront[blkpts['B']])
    self.assertEqual({blkpts['B']}, domfront[blkpts['D']])