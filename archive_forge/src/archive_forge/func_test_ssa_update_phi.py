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
def test_ssa_update_phi(self):

    @njit(pipeline_class=self.SSAPrunerCompiler)
    def impl(p=None, q=None):
        z = 1
        r = False
        if p is None:
            r = True
        if r and q is not None:
            z = 20
        return (z, r)
    self.assertPreciseEqual(impl(), impl.py_func())