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
def instruction_hoist(self):
    print('')
    print('Instruction hoisting:')
    hoist_info_printed = False
    if self.hoist_info:
        for pf_id, data in self.hoist_info.items():
            hoisted = data.get('hoisted', None)
            not_hoisted = data.get('not_hoisted', None)
            if not hoisted and (not not_hoisted):
                print('loop #%s has nothing to hoist.' % pf_id)
                continue
            print('loop #%s:' % pf_id)
            if hoisted:
                print('  Has the following hoisted:')
                [print('    %s' % y) for y in hoisted]
                hoist_info_printed = True
            if not_hoisted:
                print('  Failed to hoist the following:')
                [print('    %s: %s' % (y, x)) for x, y in not_hoisted]
                hoist_info_printed = True
    if not hoist_info_printed:
        print_wrapped('No instruction hoisting found')
    print_wrapped(80 * '-')