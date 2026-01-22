import numba
import numba.parfors.parfor
from numba import njit
from numba.core import ir_utils
from numba.core import types, ir,  compiler
from numba.core.registry import cpu_target
from numba.core.ir_utils import (copy_propagate, apply_copy_propagate,
from numba.core.typed_passes import type_inference_stage
from numba.core.compiler_machinery import FunctionPass, register_pass, PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
import numpy as np
from numba.tests.support import skip_parfors_unsupported, needs_blas
import unittest
@skip_parfors_unsupported
def test_alias_parfor_extension(self):
    """Make sure aliases are considered in remove dead extension for
        parfors.
        """

    def func():
        n = 11
        numba.parfors.parfor.init_prange()
        A = np.empty(n)
        B = A
        for i in numba.prange(n):
            A[i] = i
        return B

    @register_pass(analysis_only=False, mutates_CFG=True)
    class LimitedParfor(FunctionPass):
        _name = 'limited_parfor'

        def __init__(self):
            FunctionPass.__init__(self)

        def run_pass(self, state):
            parfor_pass = numba.parfors.parfor.ParforPass(state.func_ir, state.typemap, state.calltypes, state.return_type, state.typingctx, state.flags.auto_parallel, state.flags, state.metadata, state.parfor_diagnostics)
            remove_dels(state.func_ir.blocks)
            parfor_pass.array_analysis.run(state.func_ir.blocks)
            parfor_pass._convert_loop(state.func_ir.blocks)
            remove_dead(state.func_ir.blocks, state.func_ir.arg_names, state.func_ir, state.typemap)
            numba.parfors.parfor.get_parfor_params(state.func_ir.blocks, parfor_pass.options.fusion, parfor_pass.nested_fusion_info)
            return True

    class TestPipeline(compiler.Compiler):
        """Test pipeline that just converts prange() to parfor and calls
            remove_dead(). Copy propagation can replace B in the example code
            which this pipeline avoids.
            """

        def define_pipelines(self):
            name = 'test parfor aliasing'
            pm = PassManager(name)
            pm.add_pass(TranslateByteCode, 'analyzing bytecode')
            pm.add_pass(FixupArgs, 'fix up args')
            pm.add_pass(IRProcessing, 'processing IR')
            pm.add_pass(WithLifting, 'Handle with contexts')
            if not self.state.flags.no_rewrites:
                pm.add_pass(GenericRewrites, 'nopython rewrites')
                pm.add_pass(RewriteSemanticConstants, 'rewrite semantic constants')
                pm.add_pass(DeadBranchPrune, 'dead branch pruning')
            pm.add_pass(InlineClosureLikes, 'inline calls to locally defined closures')
            pm.add_pass(NopythonTypeInference, 'nopython frontend')
            pm.add_pass(NativeLowering, 'native lowering')
            pm.add_pass(NoPythonBackend, 'nopython mode backend')
            pm.finalize()
            return [pm]
    test_res = numba.jit(pipeline_class=TestPipeline)(func)()
    py_res = func()
    np.testing.assert_array_equal(test_res, py_res)