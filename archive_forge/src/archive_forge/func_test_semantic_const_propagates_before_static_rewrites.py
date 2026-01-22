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
def test_semantic_const_propagates_before_static_rewrites(self):

    @njit
    def impl(a, b):
        return a.shape[:b.ndim]
    args = (np.zeros((5, 4, 3, 2)), np.zeros((1, 1)))
    self.assertPreciseEqual(impl(*args), impl.py_func(*args))