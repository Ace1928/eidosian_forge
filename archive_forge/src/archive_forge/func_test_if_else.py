import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_if_else(self):

    def foo(a, b):
        c = 0
        SET_BLOCK_A
        if a < b:
            SET_BLOCK_B
            c = 1
        elif SET_BLOCK_C0:
            SET_BLOCK_C1
            c = 2
        else:
            SET_BLOCK_D
            c = 3
        SET_BLOCK_E
        if a % b == 0:
            SET_BLOCK_F
            c += 1
        SET_BLOCK_G
        return c
    cfa, blkpts = self.get_cfa_and_namedblocks(foo)
    idoms = cfa.graph.immediate_dominators()
    self.assertEqual(blkpts['A'], idoms[blkpts['B']])
    self.assertEqual(blkpts['A'], idoms[blkpts['C0']])
    self.assertEqual(blkpts['C0'], idoms[blkpts['C1']])
    self.assertEqual(blkpts['C0'], idoms[blkpts['D']])
    self.assertEqual(blkpts['A'], idoms[blkpts['E']])
    self.assertEqual(blkpts['E'], idoms[blkpts['F']])
    self.assertEqual(blkpts['E'], idoms[blkpts['G']])
    domfront = cfa.graph.dominance_frontier()
    self.assertFalse(domfront[blkpts['A']])
    self.assertFalse(domfront[blkpts['E']])
    self.assertFalse(domfront[blkpts['G']])
    self.assertEqual({blkpts['E']}, domfront[blkpts['B']])
    self.assertEqual({blkpts['E']}, domfront[blkpts['C0']])
    self.assertEqual({blkpts['E']}, domfront[blkpts['C1']])
    self.assertEqual({blkpts['E']}, domfront[blkpts['D']])
    self.assertEqual({blkpts['G']}, domfront[blkpts['F']])