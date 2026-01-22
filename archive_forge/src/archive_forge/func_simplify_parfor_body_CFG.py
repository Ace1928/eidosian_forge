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
def simplify_parfor_body_CFG(blocks):
    """simplify CFG of body loops in parfors"""
    n_parfors = 0
    for block in blocks.values():
        for stmt in block.body:
            if isinstance(stmt, Parfor):
                n_parfors += 1
                parfor = stmt
                last_block = parfor.loop_body[max(parfor.loop_body.keys())]
                scope = last_block.scope
                loc = ir.Loc('parfors_dummy', -1)
                const = ir.Var(scope, mk_unique_var('$const'), loc)
                last_block.body.append(ir.Assign(ir.Const(0, loc), const, loc))
                last_block.body.append(ir.Return(const, loc))
                parfor.loop_body = simplify_CFG(parfor.loop_body)
                last_block = parfor.loop_body[max(parfor.loop_body.keys())]
                last_block.body.pop()
                simplify_parfor_body_CFG(parfor.loop_body)
    return n_parfors