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
def has_cross_iter_dep(parfor, func_ir, typemap, index_positions=None, indexed_arrays=None, non_indexed_arrays=None):
    indices = {l.index_variable.name for l in parfor.loop_nests}
    derived_from_indices = set()
    if index_positions is None:
        index_positions = {}
    if indexed_arrays is None:
        indexed_arrays = set()
    if non_indexed_arrays is None:
        non_indexed_arrays = set()

    def add_check_position(new_position, array_accessed, index_positions, indexed_arrays, non_indexed_arrays):
        """Returns True if there is a reason to prevent fusion based
           on the rules described above.
           new_position will be a list or tuple of booleans that
           says whether the index in that spot is a parfor index
           or not.  array_accessed is the array on which the access
           is occurring."""
        if isinstance(new_position, list):
            new_position = tuple(new_position)
        if True not in new_position:
            if array_accessed in indexed_arrays:
                return True
            else:
                non_indexed_arrays.add(array_accessed)
                return False
        if array_accessed in non_indexed_arrays:
            return True
        indexed_arrays.add(array_accessed)
        npsize = len(new_position)
        if npsize not in index_positions:
            index_positions[npsize] = new_position
            return False
        return index_positions[npsize] != new_position

    def check_index(stmt_index, array_accessed, index_positions, indexed_arrays, non_indexed_arrays, derived_from_indices):
        """Looks at the indices of a getitem or setitem to see if there
           is a reason that they would prevent fusion.
           Returns True if fusion should be prohibited, False otherwise.
        """
        if isinstance(stmt_index, ir.Var):
            if isinstance(typemap[stmt_index.name], types.BaseTuple):
                fbs_res = guard(find_build_sequence, func_ir, stmt_index)
                if fbs_res is not None:
                    ind_seq, _ = fbs_res
                    if all([x.name in indices or x.name not in derived_from_indices for x in ind_seq]):
                        new_index_positions = [x.name in indices for x in ind_seq]
                        return add_check_position(new_index_positions, array_accessed, index_positions, indexed_arrays, non_indexed_arrays)
                    else:
                        return True
                else:
                    return True
            elif stmt_index.name in indices:
                return add_check_position((True,), array_accessed, index_positions, indexed_arrays, non_indexed_arrays)
            elif stmt_index.name in derived_from_indices:
                return True
            else:
                return add_check_position((False,), array_accessed, index_positions, indexed_arrays, non_indexed_arrays)
        else:
            return True
        raise errors.InternalError("Some code path in the parfor fusion cross-iteration dependency checker check_index didn't return a result.")
    for b in parfor.loop_body.values():
        for stmt in b.body:
            if isinstance(stmt, (ir.SetItem, ir.StaticSetItem)):
                if isinstance(typemap[stmt.target.name], types.npytypes.Array):
                    if check_index(stmt.index, stmt.target.name, index_positions, indexed_arrays, non_indexed_arrays, derived_from_indices):
                        return (True, index_positions, indexed_arrays, non_indexed_arrays)
                continue
            elif isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Var):
                    if stmt.value.name in indices:
                        indices.add(stmt.target.name)
                        continue
                elif isinstance(stmt.value, ir.Expr):
                    op = stmt.value.op
                    if op in ['getitem', 'static_getitem']:
                        if isinstance(typemap[stmt.value.value.name], types.npytypes.Array):
                            if check_index(stmt.value.index, stmt.value.value.name, index_positions, indexed_arrays, non_indexed_arrays, derived_from_indices):
                                return (True, index_positions, indexed_arrays, non_indexed_arrays)
                        continue
                    elif op == 'call':
                        if any([isinstance(typemap[x.name], types.npytypes.Array) for x in stmt.value.list_vars()]):
                            return (True, index_positions, indexed_arrays, non_indexed_arrays)
                    rhs_vars = [x.name for x in stmt.value.list_vars()]
                    if not indices.isdisjoint(rhs_vars) or not derived_from_indices.isdisjoint(rhs_vars):
                        derived_from_indices.add(stmt.target.name)
    return (False, index_positions, indexed_arrays, non_indexed_arrays)