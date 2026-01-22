import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_descendents(self):
    g = self.loopless2()
    d = g.descendents(34)
    self.assertEqual(sorted(d), [])
    d = g.descendents(42)
    self.assertEqual(sorted(d), [])
    d = g.descendents(21)
    self.assertEqual(sorted(d), [34, 42])
    d = g.descendents(99)
    self.assertEqual(sorted(d), [12, 18, 21, 34, 42])
    g = self.infinite_loop1()
    d = g.descendents(26)
    self.assertEqual(sorted(d), [])
    d = g.descendents(19)
    self.assertEqual(sorted(d), [])
    d = g.descendents(13)
    self.assertEqual(sorted(d), [19, 26])
    d = g.descendents(10)
    self.assertEqual(sorted(d), [13, 19, 26])
    d = g.descendents(6)
    self.assertEqual(sorted(d), [])
    d = g.descendents(0)
    self.assertEqual(sorted(d), [6, 10, 13, 19, 26])