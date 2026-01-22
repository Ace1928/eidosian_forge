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
def dead_code_elimination(func_ir, typemap=None, alias_map=None, arg_aliases=None):
    """ Performs dead code elimination and leaves the IR in a valid state on
    exit
    """
    do_post_proc = False
    while remove_dead(func_ir.blocks, func_ir.arg_names, func_ir, typemap, alias_map, arg_aliases):
        do_post_proc = True
    if do_post_proc:
        post_proc = postproc.PostProcessor(func_ir)
        post_proc.run()