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
def gen_switch(self, data, index):
    """
        Generates a function with a switch table like
        def foo():
            if PLACEHOLDER_INDEX in (<integers>):
                SENTINEL = None
            elif PLACEHOLDER_INDEX in (<integers>):
                SENTINEL = None
            ...
            else:
                raise RuntimeError

        The data is a map of (type : indexes) for example:
        (int64, int64, float64)
        might give:
        {int64: [0, 1], float64: [2]}

        The index is the index variable for the driving range loop over the
        mixed tuple.
        """
    elif_tplt = '\n\telif PLACEHOLDER_INDEX in (%s,):\n\t\tSENTINEL = None'
    b = 'def foo():\n\tif PLACEHOLDER_INDEX in (%s,):\n\t\tSENTINEL = None\n%s\n\telse:\n\t\traise RuntimeError("Unreachable")\n\tpy310_defeat1 = 1\n\tpy310_defeat2 = 2\n\tpy310_defeat3 = 3\n\tpy310_defeat4 = 4\n\t'
    keys = [k for k in data.keys()]
    elifs = []
    for i in range(1, len(keys)):
        elifs.append(elif_tplt % ','.join(map(str, data[keys[i]])))
    src = b % (','.join(map(str, data[keys[0]])), ''.join(elifs))
    wstr = src
    l = {}
    exec(wstr, {}, l)
    bfunc = l['foo']
    branches = compile_to_numba_ir(bfunc, {})
    for lbl, blk in branches.blocks.items():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Global):
                    if stmt.value.name == 'PLACEHOLDER_INDEX':
                        stmt.value = index
    return branches