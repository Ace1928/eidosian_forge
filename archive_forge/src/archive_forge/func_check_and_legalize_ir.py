import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def check_and_legalize_ir(func_ir, flags: 'numba.core.compiler.Flags'):
    """
    This checks that the IR presented is legal
    """
    enforce_no_phis(func_ir)
    enforce_no_dels(func_ir)
    post_proc = postproc.PostProcessor(func_ir)
    post_proc.run(True, extend_lifetimes=flags.dbg_extend_lifetimes)