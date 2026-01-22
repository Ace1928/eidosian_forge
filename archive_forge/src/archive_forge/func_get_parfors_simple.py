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
def get_parfors_simple(self, print_loop_search):
    parfors_simple = dict()
    for pf in sorted(self.initial_parfors, key=lambda x: x.loc.line):
        r_pattern = pf.patterns[0]
        pattern = pf.patterns[0]
        loc = pf.loc
        if isinstance(pattern, tuple):
            if pattern[0] == 'prange':
                if pattern[1] == 'internal':
                    replfn = '.'.join(reversed(list(pattern[2][0])))
                    loc = pattern[2][1]
                    r_pattern = '%s %s' % (replfn, '(internal parallel version)')
                elif pattern[1] == 'user':
                    r_pattern = 'user defined prange'
                elif pattern[1] == 'pndindex':
                    r_pattern = 'internal pndindex'
                else:
                    assert 0
        fmt = 'Parallel for-loop #%s: is produced from %s:\n    %s\n \n'
        if print_loop_search:
            print_wrapped(fmt % (pf.id, loc, r_pattern))
        parfors_simple[pf.id] = (pf, loc, r_pattern)
    return parfors_simple