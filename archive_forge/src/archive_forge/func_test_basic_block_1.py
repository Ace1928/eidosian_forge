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
def test_basic_block_1(self):

    def func(n):
        a = np.zeros(n)
        acc = 0
        if n > 4:
            b = a[1:]
            acc += b[1]
        else:
            c = a[:-1]
            acc += c[0]
        return acc
    self.check(func, types.intp, basicblock=True)