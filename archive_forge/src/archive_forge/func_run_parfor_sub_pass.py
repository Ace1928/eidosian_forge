import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
@classmethod
def run_parfor_sub_pass(cls, test_func, args):
    tp, options, diagnostics, _ = cls._run_parfor(test_func, args)
    flags = compiler.Flags()
    parfor_pass = numba.parfors.parfor.ParforPass(tp.state.func_ir, tp.state.typemap, tp.state.calltypes, tp.state.return_type, tp.state.typingctx, tp.state.targetctx, options, flags, tp.state.metadata, diagnostics=diagnostics)
    parfor_pass._pre_run()
    sub_pass = cls.sub_pass_class(parfor_pass)
    sub_pass.run(parfor_pass.func_ir.blocks)
    return sub_pass