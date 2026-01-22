import sys
import subprocess
import numpy as np
import os
import warnings
from numba import jit, njit, types
from numba.core import errors
from numba.experimental import structref
from numba.extending import (overload, intrinsic, overload_method,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
from numba.core.compiler_machinery import PassManager
from numba.core.types.functions import _err_reasons as error_reasons
from numba.tests.support import (skip_parfors_unsupported, override_config,
import unittest
def test_use_of_ir_unknown_loc(self):

    class TestPipeline(CompilerBase):

        def define_pipelines(self):
            name = 'bad_DCE_pipeline'
            pm = PassManager(name)
            pm.add_pass(TranslateByteCode, 'analyzing bytecode')
            pm.add_pass(FixupArgs, 'fix up args')
            pm.add_pass(IRProcessing, 'processing IR')
            pm.add_pass(DeadCodeElimination, 'DCE')
            pm.add_pass(NopythonTypeInference, 'nopython frontend')
            pm.add_pass(NativeLowering, 'native lowering')
            pm.add_pass(NoPythonBackend, 'nopython mode backend')
            pm.finalize()
            return [pm]

    @njit(pipeline_class=TestPipeline)
    def f(a):
        return 0
    with self.assertRaises(errors.TypingError) as raises:
        f(iter([1, 2]))
    expected = 'File "unknown location", line 0:'
    self.assertIn(expected, str(raises.exception))