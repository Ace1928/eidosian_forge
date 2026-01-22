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
def test_meminfo_refct_2(self):
    d = Dummy()
    self.assertEqual(Dummy.alive, 1)
    addr = 3735931646
    mi = rtsys.meminfo_new(addr, d)
    self.assertEqual(mi.refcount, 1)
    del d
    self.assertEqual(Dummy.alive, 1)
    for ct in range(100):
        mi.acquire()
    self.assertEqual(mi.refcount, 1 + 100)
    self.assertEqual(Dummy.alive, 1)
    for _ in range(100):
        mi.release()
    self.assertEqual(mi.refcount, 1)
    del mi
    self.assertEqual(Dummy.alive, 0)