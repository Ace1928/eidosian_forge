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
def parfor_defs(parfor, use_set=None, def_set=None):
    """list variables written in this parfor by recursively
    calling compute_use_defs() on body and combining block defs.
    """
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()
    blocks = wrap_parfor_blocks(parfor)
    uses, defs = compute_use_defs(blocks)
    cfg = compute_cfg_from_blocks(blocks)
    last_label = max(blocks.keys())
    unwrap_parfor_blocks(parfor)
    topo_order = cfg.topo_order()
    definitely_executed = cfg.dominators()[last_label]
    for loop in cfg.loops().values():
        definitely_executed -= loop.body
    for label in topo_order:
        if label in definitely_executed:
            use_set.update(uses[label] - def_set)
            def_set.update(defs[label])
        else:
            use_set.update(uses[label] - def_set)
    loop_vars = {l.start.name for l in parfor.loop_nests if isinstance(l.start, ir.Var)}
    loop_vars |= {l.stop.name for l in parfor.loop_nests if isinstance(l.stop, ir.Var)}
    loop_vars |= {l.step.name for l in parfor.loop_nests if isinstance(l.step, ir.Var)}
    use_set.update(loop_vars - def_set)
    use_set |= get_parfor_pattern_vars(parfor)
    return analysis._use_defs_result(usemap=use_set, defmap=def_set)