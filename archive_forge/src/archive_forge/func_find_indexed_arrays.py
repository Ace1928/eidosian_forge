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
def find_indexed_arrays():
    """find expressions that involve getitem using the
                        index variable. Return both the arrays and expressions.
                        """
    indices = copy.copy(loop_index_vars)
    for block in loop_body.values():
        for inst in block.find_insts(ir.Assign):
            if isinstance(inst.value, ir.Var) and inst.value.name in indices:
                indices.add(inst.target.name)
    arrs = []
    exprs = []
    for block in loop_body.values():
        for inst in block.body:
            lv = set((x.name for x in inst.list_vars()))
            if lv & indices:
                if lv.issubset(indices):
                    continue
                require(isinstance(inst, ir.Assign))
                expr = inst.value
                require(isinstance(expr, ir.Expr) and expr.op in ['getitem', 'static_getitem'])
                arrs.append(expr.value.name)
                exprs.append(expr)
    return (arrs, exprs)