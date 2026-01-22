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
def foo_ovld(dtype):
    if not isinstance(dtype, types.StringLiteral):

        def foo_noop(dtype):
            return literally(dtype)
        return foo_noop
    if dtype.literal_value == 'str':

        def foo_as_str_impl(dtype):
            return 10
        return foo_as_str_impl
    if dtype.literal_value in ('int64', 'float64'):

        def foo_as_num_impl(dtype):
            return 20
        return foo_as_num_impl