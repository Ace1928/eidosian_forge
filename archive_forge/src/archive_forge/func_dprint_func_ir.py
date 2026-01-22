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
def dprint_func_ir(func_ir, title, blocks=None):
    """Debug print function IR, with an optional blocks argument
    that may differ from the IR's original blocks.
    """
    if config.DEBUG_ARRAY_OPT >= 1:
        ir_blocks = func_ir.blocks
        func_ir.blocks = ir_blocks if blocks == None else blocks
        name = func_ir.func_id.func_qualname
        print(('IR %s: %s' % (title, name)).center(80, '-'))
        func_ir.dump()
        print('-' * 40)
        func_ir.blocks = ir_blocks