import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def double_infinite_loop(x, y):
    L = x
    i = y
    while True:
        while True:
            if i == L - 1:
                break
            i += 1
        i += 1
        if i >= L:
            break
    return (i, L)