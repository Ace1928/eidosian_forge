import sys
import numpy as np
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import import_dynamic, temp_directory
from numba.core.types import complex128
def load_ool_module():
    """
    Compile an out-of-line module, return the corresponding ffi and
    module objects.
    """
    from cffi import FFI
    numba_complex = '\n    typedef struct _numba_complex {\n        double real;\n        double imag;\n    } numba_complex;\n    '
    bool_define = '\n    #ifdef _MSC_VER\n        #define false 0\n        #define true 1\n        #define bool int\n    #else\n        #include <stdbool.h>\n    #endif\n    '
    defs = numba_complex + '\n    bool boolean(void);\n    double sin(double x);\n    double cos(double x);\n    int foo(int a, int b, int c);\n    void vsSin(int n, float* x, float* y);\n    void vdSin(int n, double* x, double* y);\n    void vector_real(numba_complex *c, double *real, int n);\n    void vector_imag(numba_complex *c, double *imag, int n);\n    '
    source = numba_complex + bool_define + '\n    static bool boolean(void)\n    {\n        return true;\n    }\n\n    static int foo(int a, int b, int c)\n    {\n        return a + b * c;\n    }\n\n    void vsSin(int n, float* x, float* y) {\n        int i;\n        for (i=0; i<n; i++)\n            y[i] = sin(x[i]);\n    }\n\n    void vdSin(int n, double* x, double* y) {\n        int i;\n        for (i=0; i<n; i++)\n            y[i] = sin(x[i]);\n    }\n\n    static void vector_real(numba_complex *c, double *real, int n) {\n        int i;\n        for (i = 0; i < n; i++)\n            real[i] = c[i].real;\n    }\n\n    static void vector_imag(numba_complex *c, double *imag, int n) {\n        int i;\n        for (i = 0; i < n; i++)\n            imag[i] = c[i].imag;\n    }\n    '
    ffi = FFI()
    ffi.set_source('cffi_usecases_ool', source)
    ffi.cdef(defs, override=True)
    tmpdir = temp_directory('test_cffi')
    ffi.compile(tmpdir=tmpdir)
    sys.path.append(tmpdir)
    try:
        mod = import_dynamic('cffi_usecases_ool')
        cffi_support.register_module(mod)
        cffi_support.register_type(mod.ffi.typeof('struct _numba_complex'), complex128)
        return (mod.ffi, mod)
    finally:
        sys.path.remove(tmpdir)