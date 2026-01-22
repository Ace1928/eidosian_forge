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
def select_template(templates, args):
    if templates is None:
        return None
    impl = None
    for template in templates:
        inline_type = getattr(template, '_inline', None)
        if inline_type is None:
            continue
        if args not in template._inline_overloads:
            continue
        if not inline_type.is_never_inline:
            try:
                impl = template._overload_func(*args)
                if impl is None:
                    raise Exception
                break
            except Exception:
                continue
    else:
        return None
    return (template, inline_type, impl)