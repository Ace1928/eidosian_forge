import sys
import numpy as np
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import import_dynamic, temp_directory
from numba.core.types import complex128
def vector_extract_real(x, y):
    vector_real(ffi.from_buffer(x), ffi.from_buffer(y), len(x))