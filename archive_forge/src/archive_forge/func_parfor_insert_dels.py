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
def parfor_insert_dels(parfor, curr_dead_set):
    """insert dels in parfor. input: dead variable set right after parfor.
    returns the variables for which del was inserted.
    """
    blocks = wrap_parfor_blocks(parfor)
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    dead_map = compute_dead_maps(cfg, blocks, live_map, usedefs.defmap)
    loop_vars = {l.start.name for l in parfor.loop_nests if isinstance(l.start, ir.Var)}
    loop_vars |= {l.stop.name for l in parfor.loop_nests if isinstance(l.stop, ir.Var)}
    loop_vars |= {l.step.name for l in parfor.loop_nests if isinstance(l.step, ir.Var)}
    loop_vars |= {l.index_variable.name for l in parfor.loop_nests}
    dead_set = set()
    for label in blocks.keys():
        dead_map.internal[label] &= curr_dead_set
        dead_map.internal[label] -= loop_vars
        dead_set |= dead_map.internal[label]
        dead_map.escaping[label] &= curr_dead_set
        dead_map.escaping[label] -= loop_vars
        dead_set |= dead_map.escaping[label]

    class DummyFuncIR(object):

        def __init__(self, blocks):
            self.blocks = blocks
    post_proc = postproc.PostProcessor(DummyFuncIR(blocks))
    post_proc._patch_var_dels(dead_map.internal, dead_map.escaping)
    unwrap_parfor_blocks(parfor)
    return dead_set | loop_vars