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
def test_single_if_const_val(self):

    def impl(x):
        if x == 100:
            return 3.14159
    self.assert_prune(impl, (types.NoneType('none'),), [True], None)
    self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)

    def impl(x):
        if 100 == x:
            return 3.14159
    self.assert_prune(impl, (types.NoneType('none'),), [True], None)
    self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)