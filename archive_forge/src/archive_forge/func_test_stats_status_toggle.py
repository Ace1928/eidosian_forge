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
def test_stats_status_toggle(self):

    @njit
    def foo():
        tmp = np.ones(3)
        return np.arange(5 * tmp[0])
    _nrt_python.memsys_enable_stats()
    self.assertTrue(_nrt_python.memsys_stats_enabled())
    for i in range(2):
        stats_1 = rtsys.get_allocation_stats()
        _nrt_python.memsys_disable_stats()
        self.assertFalse(_nrt_python.memsys_stats_enabled())
        foo()
        _nrt_python.memsys_enable_stats()
        self.assertTrue(_nrt_python.memsys_stats_enabled())
        stats_2 = rtsys.get_allocation_stats()
        foo()
        stats_3 = rtsys.get_allocation_stats()
        self.assertEqual(stats_1, stats_2)
        self.assertLess(stats_2, stats_3)