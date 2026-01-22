import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
@unittest.skip('Pass removed as it was buggy. Re-enable when fixed.')
def test_refct_pruning_with_branches(self):
    """testcase from #2350"""

    @njit
    def _append_non_na(x, y, agg, field):
        if not np.isnan(field):
            agg[y, x] += 1

    @njit
    def _append(x, y, agg, field):
        if not np.isnan(field):
            if np.isnan(agg[y, x]):
                agg[y, x] = field
            else:
                agg[y, x] += field

    @njit
    def append(x, y, agg, field):
        _append_non_na(x, y, agg, field)
        _append(x, y, agg, field)

    @njit(no_cpython_wrapper=True)
    def extend(arr, field):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                append(j, i, arr, field)
    extend.compile('(f4[:,::1], f4)')
    llvmir = str(extend.inspect_llvm(extend.signatures[0]))
    refops = list(re.finditer('(NRT_incref|NRT_decref)\\([^\\)]+\\)', llvmir))
    self.assertEqual(len(refops), 0)