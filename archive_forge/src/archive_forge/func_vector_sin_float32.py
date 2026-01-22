import sys
import numpy as np
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import import_dynamic, temp_directory
from numba.core.types import complex128
def vector_sin_float32(x, y):
    vsSin(len(x), ffi.from_buffer(x), ffi_ool.from_buffer(y))