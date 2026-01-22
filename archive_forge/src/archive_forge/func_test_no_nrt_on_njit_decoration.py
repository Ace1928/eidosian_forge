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
@TestCase.run_test_in_subprocess
def test_no_nrt_on_njit_decoration(self):
    from numba import njit
    self.assertFalse(rtsys._init)

    @njit
    def foo():
        return 123
    self.assertFalse(rtsys._init)
    self.assertEqual(foo(), foo.py_func())
    self.assertTrue(rtsys._init)