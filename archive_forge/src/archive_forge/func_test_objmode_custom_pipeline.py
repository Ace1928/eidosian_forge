from numba.core.compiler import Compiler, DefaultPassBuilder
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core.untyped_passes import InlineInlinables
from numba.core.typed_passes import IRLegalization
from numba import jit, objmode, njit, cfunc
from numba.core import types, postproc, errors
from numba.core.ir import FunctionIR
from numba.tests.support import TestCase
def test_objmode_custom_pipeline(self):
    self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [])

    @jit(pipeline_class=self.pipeline_class)
    def foo(x):
        with objmode(x='intp'):
            x += int(1)
        return x
    arg = 123
    self.assertEqual(foo(arg), arg + 1)
    self.assertEqual(len(self.pipeline_class.custom_pipeline_cache), 2)
    first = self.pipeline_class.custom_pipeline_cache[0]
    self.assertIs(first, foo.py_func)
    second = self.pipeline_class.custom_pipeline_cache[1]
    self.assertIsInstance(second, FunctionIR)