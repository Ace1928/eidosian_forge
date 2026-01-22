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
def test_fanout_3(self):

    def func(n):
        ary = np.arange(n)
        c = 0
        for v in np.nditer(ary):
            c += v.item()
        return 1
    with set_refprune_flags('per_bb,fanout_raise'):
        self.check(func, types.intp, basicblock=True, diamond=False, fanout=False, fanout_raise=True)