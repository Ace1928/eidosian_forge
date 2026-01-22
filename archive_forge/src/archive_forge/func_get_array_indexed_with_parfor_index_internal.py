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
def get_array_indexed_with_parfor_index_internal(loop_body, index, ret_indexed, ret_not_indexed, nest_indices, func_ir):
    for blk in loop_body:
        for stmt in blk.body:
            if isinstance(stmt, (ir.StaticSetItem, ir.SetItem)):
                setarray_index = get_index_var(stmt)
                if isinstance(setarray_index, ir.Var) and (setarray_index.name == index or _is_indirect_index(func_ir, setarray_index, nest_indices)):
                    ret_indexed.add(stmt.target.name)
                else:
                    ret_not_indexed.add(stmt.target.name)
            elif isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op in ['getitem', 'static_getitem']):
                getarray_index = stmt.value.index
                getarray_name = stmt.value.value.name
                if isinstance(getarray_index, ir.Var) and (getarray_index.name == index or _is_indirect_index(func_ir, getarray_index, nest_indices)):
                    ret_indexed.add(getarray_name)
                else:
                    ret_not_indexed.add(getarray_name)
            elif isinstance(stmt, Parfor):
                get_array_indexed_with_parfor_index_internal(stmt.loop_body.values(), index, ret_indexed, ret_not_indexed, nest_indices, func_ir)