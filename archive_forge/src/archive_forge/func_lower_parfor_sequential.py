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
def lower_parfor_sequential(typingctx, func_ir, typemap, calltypes, metadata):
    ir_utils._the_max_label.update(ir_utils.find_max_label(func_ir.blocks))
    parfor_found = False
    new_blocks = {}
    scope = next(iter(func_ir.blocks.values())).scope
    for block_label, block in func_ir.blocks.items():
        block_label, parfor_found = _lower_parfor_sequential_block(block_label, block, new_blocks, typemap, calltypes, parfor_found, scope=scope)
        new_blocks[block_label] = block
    func_ir.blocks = new_blocks
    if parfor_found:
        func_ir.blocks = rename_labels(func_ir.blocks)
    dprint_func_ir(func_ir, 'after parfor sequential lowering')
    simplify(func_ir, typemap, calltypes, metadata['parfors'])
    dprint_func_ir(func_ir, 'after parfor sequential simplify')