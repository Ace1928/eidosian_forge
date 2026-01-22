import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
@overload(output_factory, inline='always')
def ol_output_factory(array, dtype):
    if isinstance(array, types.npytypes.Array):

        def impl(array, dtype):
            shape = array.shape[3:]
            return np.zeros(shape, dtype=dtype)
        return impl