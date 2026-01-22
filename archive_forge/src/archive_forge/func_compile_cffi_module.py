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
def compile_cffi_module(self, name, source, cdef):
    from cffi import FFI
    ffi = FFI()
    ffi.set_source(name, source, include_dirs=[include_path()])
    ffi.cdef(cdef)
    tmpdir = temp_directory('cffi_test_{}'.format(name))
    ffi.compile(tmpdir=tmpdir)
    sys.path.append(tmpdir)
    try:
        mod = import_dynamic(name)
    finally:
        sys.path.remove(tmpdir)
    return (ffi, mod)