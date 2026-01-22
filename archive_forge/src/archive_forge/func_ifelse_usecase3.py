import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def ifelse_usecase3(x, y):
    if x > 0:
        if y > 0:
            return 1
        elif y < 0:
            return 1
        else:
            return 0
    elif x < 0:
        return 1
    else:
        return 0