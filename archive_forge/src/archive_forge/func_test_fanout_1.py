import unittest
import warnings
from contextlib import contextmanager
import numpy as np
import llvmlite.binding as llvm
from numba import njit, types
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.codegen import _parse_refprune_flags
from numba.tests.support import override_config, TestCase
@TestCase.run_test_in_subprocess
def test_fanout_1(self):

    def func(n):
        a = np.zeros(n)
        b = np.zeros(n)
        x = (a, b)
        acc = 0.0
        for i in x:
            acc += i[0]
        return acc
    self.check(func, types.intp, basicblock=True, fanout=True)