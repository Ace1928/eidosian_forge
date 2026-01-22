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
def test_refct_pruning_issue_1526(self):

    @njit
    def udt(image, x, y):
        next_loc = np.where(image == 1)
        if len(next_loc[0]) == 0:
            y_offset = 1
            x_offset = 1
        else:
            y_offset = next_loc[0][0]
            x_offset = next_loc[1][0]
        next_loc_x = x - 1 + x_offset
        next_loc_y = y - 1 + y_offset
        return (next_loc_x, next_loc_y)
    a = np.array([[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]])
    expect = udt.py_func(a, 1, 6)
    got = udt(a, 1, 6)
    self.assertEqual(expect, got)