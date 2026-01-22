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
def unwrap_parfor_blocks(parfor, blocks=None):
    """
    unwrap parfor blocks after analysis/optimization.
    Allows changes to the parfor loop.
    """
    if blocks is not None:
        init_block_label = min(blocks.keys())
        blocks.pop(init_block_label)
        parfor.loop_body = blocks
    first_body_label = min(parfor.loop_body.keys())
    assert isinstance(parfor.init_block.body[-1], ir.Jump)
    parfor.init_block.body.pop()
    for block in parfor.loop_body.values():
        if isinstance(block.body[-1], ir.Jump) and block.body[-1].target == first_body_label:
            block.body.pop()
    return