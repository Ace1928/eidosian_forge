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
def type_inference_stage(typingctx, targetctx, interp, args, return_type, locals={}, raise_errors=True):
    if len(args) != interp.arg_count:
        raise TypeError('Mismatch number of argument types')
    warnings = errors.WarningsFixer(errors.NumbaWarning)
    infer = typeinfer.TypeInferer(typingctx, interp, warnings)
    callstack_ctx = typingctx.callstack.register(targetctx.target, infer, interp.func_id, args)
    with callstack_ctx, warnings:
        for index, (name, ty) in enumerate(zip(interp.arg_names, args)):
            infer.seed_argument(name, index, ty)
        if return_type is not None:
            infer.seed_return(return_type)
        for k, v in locals.items():
            infer.seed_type(k, v)
        infer.build_constraint()
        errs = infer.propagate(raise_errors=raise_errors)
        typemap, restype, calltypes = infer.unify(raise_errors=raise_errors)
    return _TypingResults(typemap, restype, calltypes, errs)