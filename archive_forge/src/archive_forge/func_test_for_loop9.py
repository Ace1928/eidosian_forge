import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_for_loop9(self, flags=enable_pyobj_flags):
    self.run_test(for_loop_usecase9, [0, 1], [0, 2, 10], flags=flags)