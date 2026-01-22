import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def check_dominators(self, got, expected):
    self.assertEqual(sorted(got), sorted(expected))
    for node in sorted(got):
        self.assertEqual(sorted(got[node]), sorted(expected[node]), 'mismatch for %r' % (node,))