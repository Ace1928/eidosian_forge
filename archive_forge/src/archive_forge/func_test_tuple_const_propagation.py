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
def test_tuple_const_propagation(self):

    @njit(pipeline_class=IRPreservingTestPipeline)
    def impl(*args):
        s = 0
        for arg in literal_unroll(args):
            s += len(arg)
        return s
    inp = ((), (1, 2, 3), ())
    self.assertPreciseEqual(impl(*inp), impl.py_func(*inp))
    ol = impl.overloads[impl.signatures[0]]
    func_ir = ol.metadata['preserved_ir']
    binop_consts = set()
    for blk in func_ir.blocks.values():
        for expr in blk.find_exprs('inplace_binop'):
            inst = blk.find_variable_assignment(expr.rhs.name)
            self.assertIsInstance(inst.value, ir.Const)
            binop_consts.add(inst.value.value)
    self.assertEqual(binop_consts, {len(x) for x in inp})