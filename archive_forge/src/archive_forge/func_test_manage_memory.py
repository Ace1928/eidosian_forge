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
def test_manage_memory(self):
    name = '{}_test_manage_memory'.format(self.__class__.__name__)
    source = '\n#include <stdio.h>\n#include "numba/core/runtime/nrt_external.h"\n\nint status = 0;\n\nvoid my_dtor(void *ptr) {\n    free(ptr);\n    status = 0xdead;\n}\n\nNRT_MemInfo* test_nrt_api(NRT_api_functions *nrt) {\n    void * data = malloc(10);\n    NRT_MemInfo *mi = nrt->manage_memory(data, my_dtor);\n    nrt->acquire(mi);\n    nrt->release(mi);\n    status = 0xa110c;\n    return mi;\n}\n        '
    cdef = '\nvoid* test_nrt_api(void *nrt);\nextern int status;\n        '
    ffi, mod = self.compile_cffi_module(name, source, cdef)
    self.assertEqual(mod.lib.status, 0)
    table = self.get_nrt_api_table()
    out = mod.lib.test_nrt_api(table)
    self.assertEqual(mod.lib.status, 659724)
    mi_addr = int(ffi.cast('size_t', out))
    mi = nrt.MemInfo(mi_addr)
    self.assertEqual(mi.refcount, 1)
    del mi
    self.assertEqual(mod.lib.status, 57005)