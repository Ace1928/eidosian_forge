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
def print_unoptimised(self, lines):
    sword = '+--'
    fac = len(sword)
    fadj, froots = self.compute_graph_info(self.fusion_info)
    nadj, _nroots = self.compute_graph_info(self.nested_fusion_info)
    if len(fadj) > len(nadj):
        lim = len(fadj)
        tmp = nadj
    else:
        lim = len(nadj)
        tmp = fadj
    for x in range(len(tmp), lim):
        tmp.append([])

    def print_nest(fadj_, nadj_, theroot, reported, region_id):

        def print_g(fadj_, nadj_, nroot, depth):
            print_wrapped(fac * depth * ' ' + '%s%s %s' % (sword, nroot, '(parallel)'))
            for k in nadj_[nroot]:
                if nadj_[k] == []:
                    msg = []
                    msg.append(fac * (depth + 1) * ' ' + '%s%s %s' % (sword, k, '(parallel)'))
                    if fadj_[k] != [] and k not in reported:
                        fused = self.reachable_nodes(fadj_, k)
                        for i in fused:
                            msg.append(fac * (depth + 1) * ' ' + '%s%s %s' % (sword, i, '(parallel)'))
                    reported.append(k)
                    print_wrapped('\n'.join(msg))
                else:
                    print_g(fadj_, nadj_, k, depth + 1)
        if nadj_[theroot] != []:
            print_wrapped('Parallel region %s:' % region_id)
            print_g(fadj_, nadj_, theroot, 0)
            print('\n')
            region_id = region_id + 1
        return region_id

    def print_fuse(ty, pf_id, adj, depth, region_id):
        msg = []
        print_wrapped('Parallel region %s:' % region_id)
        msg.append(fac * depth * ' ' + '%s%s %s' % (sword, pf_id, '(parallel)'))
        if adj[pf_id] != []:
            fused = sorted(self.reachable_nodes(adj, pf_id))
            for k in fused:
                msg.append(fac * depth * ' ' + '%s%s %s' % (sword, k, '(parallel)'))
        region_id = region_id + 1
        print_wrapped('\n'.join(msg))
        print('\n')
        return region_id
    region_id = 0
    reported = []
    for line, info in sorted(lines.items()):
        opt_ty, pf_id, adj = info
        if opt_ty == 'fuse':
            if pf_id not in reported:
                region_id = print_fuse('f', pf_id, adj, 0, region_id)
        elif opt_ty == 'nest':
            region_id = print_nest(fadj, nadj, pf_id, reported, region_id)
        else:
            assert 0