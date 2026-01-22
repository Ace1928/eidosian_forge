import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def get_cfa_and_namedblocks(self, fn):
    fid = FunctionIdentity.from_function(fn)
    bc = ByteCode(func_id=fid)
    cfa = self.cfa(bc)
    namedblocks = self._scan_namedblocks(bc, cfa)
    return (cfa, namedblocks)