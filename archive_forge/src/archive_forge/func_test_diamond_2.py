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
def test_diamond_2(self):

    def func(n):
        con = []
        for i in range(n):
            con.append(np.arange(i))
        c = 0.0
        for arr in con:
            c += arr.sum() / (1 + arr.size)
        return c
    with set_refprune_flags('per_bb,diamond'):
        self.check(func, types.intp, basicblock=True, diamond=True, fanout=False, fanout_raise=False)