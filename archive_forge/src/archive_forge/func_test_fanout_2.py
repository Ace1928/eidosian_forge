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
def test_fanout_2(self):

    def func(n):
        a = np.zeros(n)
        b = np.zeros(n)
        x = (a, b)
        for i in x:
            if n:
                raise ValueError
        return x
    with set_refprune_flags('per_bb,fanout'):
        self.check(func, types.intp, basicblock=True, diamond=False, fanout=True, fanout_raise=False)