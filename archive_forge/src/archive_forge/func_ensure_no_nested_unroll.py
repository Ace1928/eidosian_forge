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
def ensure_no_nested_unroll(unroll_loops):
    for test_loop in unroll_loops:
        for ref_loop in unroll_loops:
            if test_loop == ref_loop:
                continue
            if test_loop.header in ref_loop.body:
                msg = 'Nesting of literal_unroll is unsupported'
                loc = func_ir.blocks[test_loop.header].loc
                raise errors.UnsupportedError(msg, loc)