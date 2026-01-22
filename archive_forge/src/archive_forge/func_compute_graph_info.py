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
def compute_graph_info(self, _a):
    """
        compute adjacency list of the fused loops
        and find the roots in of the lists
        """
    a = copy.deepcopy(_a)
    if a == {}:
        return ([], set())
    vtx = set()
    for v in a.values():
        for x in v:
            vtx.add(x)
    potential_roots = set(a.keys())
    roots = potential_roots - vtx
    if roots is None:
        roots = set()
    not_roots = set()
    for x in range(max(set(a.keys()).union(vtx)) + 1):
        val = a.get(x)
        if val is not None:
            a[x] = val
        elif val == []:
            not_roots.add(x)
        else:
            a[x] = []
    l = []
    for x in sorted(a.keys()):
        l.append(a[x])
    return (l, roots)