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
def test_alias_func_ext(self):

    def func(A, i):
        B = dummy_aliased_func(A)
        B[i, 0] = 3
    old_ext_handlers = alias_func_extensions.copy()
    try:
        alias_func_extensions['dummy_aliased_func', 'numba.tests.test_remove_dead'] = alias_ext_dummy_func
        self.run_array_index_test(func)
    finally:
        ir_utils.alias_func_extensions = old_ext_handlers