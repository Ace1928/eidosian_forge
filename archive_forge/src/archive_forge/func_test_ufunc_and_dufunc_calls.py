import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def test_ufunc_and_dufunc_calls(self):
    """
        Verify that ufunc and DUFunc calls are being properly included in
        array expressions.
        """
    A = np.random.random(10)
    B = np.random.random(10)
    arg_tys = [typeof(arg) for arg in (A, B)]
    vaxy_descr = vaxy._dispatcher.targetdescr
    control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys, typing_context=vaxy_descr.typing_context, target_context=vaxy_descr.target_context)
    cres_0 = control_pipeline.compile_extra(call_stuff)
    nb_call_stuff_0 = cres_0.entry_point
    test_pipeline = RewritesTester.mk_pipeline(arg_tys, typing_context=vaxy_descr.typing_context, target_context=vaxy_descr.target_context)
    cres_1 = test_pipeline.compile_extra(call_stuff)
    nb_call_stuff_1 = cres_1.entry_point
    expected = call_stuff(A, B)
    control = nb_call_stuff_0(A, B)
    actual = nb_call_stuff_1(A, B)
    np.testing.assert_array_almost_equal(expected, control)
    np.testing.assert_array_almost_equal(expected, actual)
    self._assert_total_rewrite(control_pipeline.state.func_ir.blocks, test_pipeline.state.func_ir.blocks)