from numba.core.compiler import Compiler, DefaultPassBuilder
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core.untyped_passes import InlineInlinables
from numba.core.typed_passes import IRLegalization
from numba import jit, objmode, njit, cfunc
from numba.core import types, postproc, errors
from numba.core.ir import FunctionIR
from numba.tests.support import TestCase
def test_no_compiler_error_on_ir_del_after_legalization(self):
    new_compiler = self._create_pipeline_w_del(AnalysisPass, IRLegalization)

    @njit(pipeline_class=new_compiler)
    def foo(x):
        return x + 1
    self.assertTrue(foo(10), foo.py_func(10))