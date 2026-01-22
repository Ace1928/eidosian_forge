import numba
import numba.parfors.parfor
from numba import njit
from numba.core import ir_utils
from numba.core import types, ir,  compiler
from numba.core.registry import cpu_target
from numba.core.ir_utils import (copy_propagate, apply_copy_propagate,
from numba.core.typed_passes import type_inference_stage
from numba.core.compiler_machinery import FunctionPass, register_pass, PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
import numpy as np
from numba.tests.support import skip_parfors_unsupported, needs_blas
import unittest
@skip_parfors_unsupported
@needs_blas
def test_alias_ctypes(self):
    from numba.np.linalg import _BLAS
    xxnrm2 = _BLAS().numba_xxnrm2(types.float64)

    def remove_dead_xxnrm2(rhs, lives, call_list):
        if call_list == [xxnrm2]:
            return rhs.args[4].name not in lives
        return False
    old_remove_handlers = remove_call_handlers[:]
    remove_call_handlers.append(remove_dead_xxnrm2)

    def func(ret):
        a = np.ones(4)
        xxnrm2(100, 4, a.ctypes, 1, ret.ctypes)
    A1 = np.zeros(1)
    A2 = A1.copy()
    try:
        pfunc = self.compile_parallel(func, (numba.typeof(A1),))
        numba.njit(func)(A1)
        pfunc(A2)
    finally:
        remove_call_handlers[:] = old_remove_handlers
    self.assertEqual(A1[0], A2[0])