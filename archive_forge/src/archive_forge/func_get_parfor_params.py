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
def get_parfor_params(blocks, options_fusion, fusion_info):
    """find variables used in body of parfors from outside and save them.
    computed as live variables at entry of first block.
    """
    parfor_ids = set()
    parfors = []
    pre_defs = set()
    _, all_defs = compute_use_defs(blocks)
    topo_order = find_topo_order(blocks)
    for label in topo_order:
        block = blocks[label]
        for i, parfor in _find_parfors(block.body):
            dummy_block = ir.Block(block.scope, block.loc)
            dummy_block.body = block.body[:i]
            before_defs = compute_use_defs({0: dummy_block}).defmap[0]
            pre_defs |= before_defs
            params = get_parfor_params_inner(parfor, pre_defs, options_fusion, fusion_info)
            parfor.params, parfor.races = _combine_params_races_for_ssa_names(block.scope, params, parfor.races)
            parfor_ids.add(parfor.id)
            parfors.append(parfor)
        pre_defs |= all_defs[label]
    return (parfor_ids, parfors)