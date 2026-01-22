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
def push_call_vars(blocks, saved_globals, saved_getattrs, typemap, nested=False):
    """push call variables to right before their call site.
    assuming one global/getattr is created for each call site and control flow
    doesn't change it.
    """
    for block in blocks.values():
        new_body = []
        block_defs = set()
        rename_dict = {}
        for stmt in block.body:

            def process_assign(stmt):
                if isinstance(stmt, ir.Assign):
                    rhs = stmt.value
                    lhs = stmt.target
                    if isinstance(rhs, ir.Global):
                        saved_globals[lhs.name] = stmt
                        block_defs.add(lhs.name)
                    elif isinstance(rhs, ir.Expr) and rhs.op == 'getattr':
                        if rhs.value.name in saved_globals or rhs.value.name in saved_getattrs:
                            saved_getattrs[lhs.name] = stmt
                            block_defs.add(lhs.name)
            if not nested and isinstance(stmt, Parfor):
                for s in stmt.init_block.body:
                    process_assign(s)
                pblocks = stmt.loop_body.copy()
                push_call_vars(pblocks, saved_globals, saved_getattrs, typemap, nested=True)
                new_body.append(stmt)
                continue
            else:
                process_assign(stmt)
            for v in stmt.list_vars():
                new_body += _get_saved_call_nodes(v.name, saved_globals, saved_getattrs, block_defs, rename_dict)
            new_body.append(stmt)
        block.body = new_body
        if len(rename_dict) > 0:
            for k, v in rename_dict.items():
                typemap[v] = typemap[k]
            temp_blocks = {0: block}
            replace_var_names(temp_blocks, rename_dict)
    return