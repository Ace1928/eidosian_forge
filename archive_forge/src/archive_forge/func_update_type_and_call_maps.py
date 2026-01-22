import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def update_type_and_call_maps(self, callee_ir, arg_typs):
    """ Updates the type and call maps based on calling callee_ir with
        arguments from arg_typs"""
    from numba.core.ssa import reconstruct_ssa
    from numba.core.typed_passes import PreLowerStripPhis
    if not self._permit_update_type_and_call_maps:
        msg = 'InlineWorker instance not configured correctly, typemap or calltypes missing in initialization.'
        raise ValueError(msg)
    from numba.core import typed_passes
    callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
    numba.core.analysis.dead_branch_prune(callee_ir, arg_typs)
    callee_ir = reconstruct_ssa(callee_ir)
    callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
    [f_typemap, _f_return_type, f_calltypes, _] = typed_passes.type_inference_stage(self.typingctx, self.targetctx, callee_ir, arg_typs, None)
    callee_ir = PreLowerStripPhis()._strip_phi_nodes(callee_ir)
    callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
    canonicalize_array_math(callee_ir, f_typemap, f_calltypes, self.typingctx)
    arg_names = [vname for vname in f_typemap if vname.startswith('arg.')]
    for a in arg_names:
        f_typemap.pop(a)
    self.typemap.update(f_typemap)
    self.calltypes.update(f_calltypes)