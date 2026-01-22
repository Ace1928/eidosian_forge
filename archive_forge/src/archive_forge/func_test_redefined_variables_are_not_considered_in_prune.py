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
def test_redefined_variables_are_not_considered_in_prune(self):

    def impl(array, a=None):
        if a is None:
            a = 0
        if a < 0:
            return 10
        return 30
    self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.NoneType('none')), [None, None], np.zeros((2, 3)), None)