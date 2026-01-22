import sys
import numpy as np
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import import_dynamic, temp_directory
from numba.core.types import complex128
def load_inline_module():
    """
    Create an inline module, return the corresponding ffi and dll objects.
    """
    from cffi import FFI
    defs = '\n    double _numba_test_sin(double x);\n    double _numba_test_cos(double x);\n    double _numba_test_funcptr(double (*func)(double));\n    bool _numba_test_boolean(void);\n    '
    ffi = FFI()
    ffi.cdef(defs)
    from numba import _helperlib
    return (ffi, ffi.dlopen(_helperlib.__file__))