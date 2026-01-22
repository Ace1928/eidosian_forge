from numba.core.compiler import Compiler, DefaultPassBuilder
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core.untyped_passes import InlineInlinables
from numba.core.typed_passes import IRLegalization
from numba import jit, objmode, njit, cfunc
from numba.core import types, postproc, errors
from numba.core.ir import FunctionIR
from numba.tests.support import TestCase
def test_cfunc_custom_pipeline(self):
    self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [])

    @cfunc(types.int64(types.int64), pipeline_class=self.pipeline_class)
    def foo(x):
        return x
    self.assertEqual(foo(4), 4)
    self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [foo.__wrapped__])