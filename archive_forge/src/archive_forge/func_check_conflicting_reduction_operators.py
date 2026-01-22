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
def check_conflicting_reduction_operators(param, nodes):
    """In prange, a user could theoretically specify conflicting
       reduction operators.  For example, in one spot it is += and
       another spot *=.  Here, we raise an exception if multiple
       different reduction operators are used in one prange.
    """
    first_red_func = None
    for node in nodes:
        if isinstance(node, ir.Assign) and isinstance(node.value, ir.Expr) and (node.value.op == 'inplace_binop'):
            if first_red_func is None:
                first_red_func = node.value.fn
            elif first_red_func != node.value.fn:
                msg = 'Reduction variable %s has multiple conflicting reduction operators.' % param.unversioned_name
                raise errors.UnsupportedRewriteError(msg, node.loc)