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
def test_init_fail(self):
    methods = {'library': (), 'meminfo_new': ((), ()), 'meminfo_alloc': ((),)}
    for meth, args in methods.items():
        try:
            with self.assertRaises(RuntimeError) as raises:
                rtsys._init = False
                fn = getattr(rtsys, meth)
                fn(*args)
            msg = 'Runtime must be initialized before use.'
            self.assertIn(msg, str(raises.exception))
        finally:
            rtsys._init = True