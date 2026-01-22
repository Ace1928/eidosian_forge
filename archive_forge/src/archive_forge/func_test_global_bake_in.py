import collections
import types as pytypes
import numpy as np
from numba.core.compiler import run_frontend, Flags, StateDict
from numba import jit, njit, literal_unroll
from numba.core import types, errors, ir, rewrites, ir_utils, utils, cpu
from numba.core import postproc
from numba.core.inline_closurecall import InlineClosureCallPass
from numba.tests.support import (TestCase, MemoryLeakMixin, SerialMixin,
from numba.core.analysis import dead_branch_prune, rewrite_semantic_constants
from numba.core.untyped_passes import (ReconstructSSA, TranslateByteCode,
from numba.core.compiler import DefaultPassBuilder, CompilerBase, PassManager
def test_global_bake_in(self):

    def impl(x):
        if _GLOBAL == 123:
            return x
        else:
            return x + 10
    self.assert_prune(impl, (types.IntegerLiteral(1),), [False], 1)
    global _GLOBAL
    tmp = _GLOBAL
    try:
        _GLOBAL = 5

        def impl(x):
            if _GLOBAL == 123:
                return x
            else:
                return x + 10
        self.assert_prune(impl, (types.IntegerLiteral(1),), [True], 1)
    finally:
        _GLOBAL = tmp