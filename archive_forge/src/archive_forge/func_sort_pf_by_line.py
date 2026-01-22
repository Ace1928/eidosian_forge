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
def sort_pf_by_line(self, pf_id, parfors_simple):
    """
        pd_id - the parfors id
        parfors_simple - the simple parfors map
        """
    pf = parfors_simple[pf_id][0]
    pattern = pf.patterns[0]
    line = max(0, pf.loc.line - 1)
    filename = self.func_ir.loc.filename
    nadj, nroots = self.compute_graph_info(self.nested_fusion_info)
    fadj, froots = self.compute_graph_info(self.fusion_info)
    graphs = [nadj, fadj]
    if isinstance(pattern, tuple):
        if pattern[1] == 'internal':
            reported_loc = pattern[2][1]
            if reported_loc.filename == filename:
                return max(0, reported_loc.line - 1)
            else:
                tmp = []
                for adj in graphs:
                    if adj:
                        for k in adj[pf_id]:
                            tmp.append(self.sort_pf_by_line(k, parfors_simple))
                        if tmp:
                            return max(0, min(tmp) - 1)
                for blk in pf.loop_body.values():
                    for stmt in blk.body:
                        if stmt.loc.filename == filename:
                            return max(0, stmt.loc.line - 1)
                for blk in self.func_ir.blocks.values():
                    try:
                        idx = blk.body.index(pf)
                        for i in range(idx - 1, 0, -1):
                            stmt = blk.body[i]
                            if not isinstance(stmt, Parfor):
                                line = max(0, stmt.loc.line - 1)
                                break
                    except ValueError:
                        pass
    return line