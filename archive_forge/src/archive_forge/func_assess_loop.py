from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import deepcopy, copy
import warnings
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core import (errors, types, ir, bytecode, postproc, rewrites, config,
from numba.misc.special import literal_unroll
from numba.core.analysis import (dead_branch_prune, rewrite_semantic_constants,
from numba.core.ir_utils import (guard, resolve_func_from_module, simplify_CFG,
from numba.core.ssa import reconstruct_ssa
from numba.core import interpreter
def assess_loop(self, loop, func_ir, partial_typemap=None):
    iternexts = [_ for _ in func_ir.blocks[loop.header].find_exprs('iternext')]
    if len(iternexts) != 1:
        return False
    for iternext in iternexts:
        phi = guard(get_definition, func_ir, iternext.value)
        if phi is None:
            return False
        if getattr(phi, 'op', False) == 'getiter':
            if partial_typemap:
                phi_val_defn = guard(get_definition, func_ir, phi.value)
                if not isinstance(phi_val_defn, ir.Expr):
                    return False
                if not phi_val_defn.op == 'call':
                    return False
                call = guard(get_definition, func_ir, phi_val_defn)
                if call is None or len(call.args) != 1:
                    return False
                func_var = guard(get_definition, func_ir, call.func)
                func = guard(get_definition, func_ir, func_var)
                if func is None or not isinstance(func, (ir.Global, ir.FreeVar)):
                    return False
                if func.value is None or func.value not in self._accepted_calls:
                    return False
                ty = partial_typemap.get(call.args[0].name, None)
                if ty and isinstance(ty, self._accepted_types):
                    return len(loop.entries) == 1
            else:
                return len(loop.entries) == 1