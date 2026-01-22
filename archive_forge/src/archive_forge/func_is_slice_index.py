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
def is_slice_index(index):
    """see if index is a slice index or has slice in it"""
    if isinstance(index, slice):
        return True
    if isinstance(index, tuple):
        for i in index:
            if isinstance(i, slice):
                return True
    return False