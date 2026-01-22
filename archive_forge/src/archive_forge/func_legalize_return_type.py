import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from functools import partial
from copy import copy
import warnings
from numba.core import (errors, types, typing, ir, funcdesc, rewrites,
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (raise_on_unsupported_feature, warn_deprecated,
from numba.core import postproc
from llvmlite import binding as llvm
def legalize_return_type(return_type, interp, targetctx):
    """
            Only accept array return type iff it is passed into the function.
            Reject function object return types if in nopython mode.
            """
    if not targetctx.enable_nrt and isinstance(return_type, types.Array):
        retstmts = []
        caststmts = {}
        argvars = set()
        for bid, blk in interp.blocks.items():
            for inst in blk.body:
                if isinstance(inst, ir.Return):
                    retstmts.append(inst.value.name)
                elif isinstance(inst, ir.Assign):
                    if isinstance(inst.value, ir.Expr) and inst.value.op == 'cast':
                        caststmts[inst.target.name] = inst.value
                    elif isinstance(inst.value, ir.Arg):
                        argvars.add(inst.target.name)
        assert retstmts, 'No return statements?'
        for var in retstmts:
            cast = caststmts.get(var)
            if cast is None or cast.value.name not in argvars:
                if self._raise_errors:
                    msg = 'Only accept returning of array passed into the function as argument'
                    raise errors.NumbaTypeError(msg)
    elif isinstance(return_type, types.Function) or isinstance(return_type, types.Phantom):
        if self._raise_errors:
            msg = "Can't return function object ({}) in nopython mode"
            raise errors.NumbaTypeError(msg.format(return_type))