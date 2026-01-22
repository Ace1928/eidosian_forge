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
def test_ssa_replace_phi(self):

    @njit(pipeline_class=self.SSAPrunerCompiler)
    def impl(p=None):
        z = 0
        if p is None:
            z = 10
        else:
            z = 20
        return z
    self.assertPreciseEqual(impl(), impl.py_func())
    func_ir = impl.overloads[impl.signatures[0]].metadata['preserved_ir']
    for blk in func_ir.blocks.values():
        self.assertFalse([*blk.find_exprs('phi')])