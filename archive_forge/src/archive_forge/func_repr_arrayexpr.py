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
def repr_arrayexpr(arrayexpr):
    """Extract operators from arrayexpr to represent it abstractly as a string.
    """
    if isinstance(arrayexpr, tuple):
        opr = arrayexpr[0]
        if not isinstance(opr, str):
            if hasattr(opr, '__name__'):
                opr = opr.__name__
            else:
                opr = '_'
        args = arrayexpr[1]
        if len(args) == 1:
            return '({}({}))'.format(opr, repr_arrayexpr(args[0]))
        else:
            opr = ' ' + opr + ' '
            return '({})'.format(opr.join([repr_arrayexpr(x) for x in args]))
    elif isinstance(arrayexpr, numba.core.ir.Var):
        name = arrayexpr.name
        if name.startswith('$'):
            return "'%s' (temporary variable)" % name
        else:
            return name
    elif isinstance(arrayexpr, numba.core.ir.Const):
        return repr(arrayexpr.value)
    else:
        return '_'