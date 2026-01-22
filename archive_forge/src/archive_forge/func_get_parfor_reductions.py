import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
def get_parfor_reductions(func_ir, parfor, parfor_params, calltypes, reductions=None, reduce_varnames=None, param_uses=None, param_nodes=None, var_to_param=None):
    """find variables that are updated using their previous values and an array
    item accessed with parfor index, e.g. s = s+A[i]
    """
    if reductions is None:
        reductions = {}
    if reduce_varnames is None:
        reduce_varnames = []
    if param_uses is None:
        param_uses = defaultdict(list)
    if param_nodes is None:
        param_nodes = defaultdict(list)
    if var_to_param is None:
        var_to_param = {}
    blocks = wrap_parfor_blocks(parfor)
    topo_order = find_topo_order(blocks)
    topo_order = topo_order[1:]
    unwrap_parfor_blocks(parfor)
    for label in reversed(topo_order):
        for stmt in reversed(parfor.loop_body[label].body):
            if isinstance(stmt, ir.Assign) and (stmt.target.name in parfor_params or stmt.target.name in var_to_param):
                lhs = stmt.target
                rhs = stmt.value
                cur_param = lhs if lhs.name in parfor_params else var_to_param[lhs.name]
                used_vars = []
                if isinstance(rhs, ir.Var):
                    used_vars = [rhs.name]
                elif isinstance(rhs, ir.Expr):
                    used_vars = [v.name for v in stmt.value.list_vars()]
                param_uses[cur_param].extend(used_vars)
                for v in used_vars:
                    var_to_param[v] = cur_param
                stmt_cp = copy.deepcopy(stmt)
                if stmt.value in calltypes:
                    calltypes[stmt_cp.value] = calltypes[stmt.value]
                param_nodes[cur_param].append(stmt_cp)
            if isinstance(stmt, Parfor):
                get_parfor_reductions(func_ir, stmt, parfor_params, calltypes, reductions, reduce_varnames, None, param_nodes, var_to_param)
    for param, used_vars in param_uses.items():
        param_name = param.name
        if param_name in used_vars and param_name not in reduce_varnames:
            param_nodes[param].reverse()
            reduce_nodes = get_reduce_nodes(param, param_nodes[param], func_ir)
            if reduce_nodes is not None:
                reduce_varnames.append(param_name)
                check_conflicting_reduction_operators(param, reduce_nodes)
                gri_out = guard(get_reduction_init, reduce_nodes)
                if gri_out is not None:
                    init_val, redop = gri_out
                else:
                    init_val = None
                    redop = None
                reductions[param_name] = _RedVarInfo(init_val=init_val, reduce_nodes=reduce_nodes, redop=redop)
    return (reduce_varnames, reductions)