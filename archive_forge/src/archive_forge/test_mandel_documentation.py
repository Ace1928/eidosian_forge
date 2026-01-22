from numba import float64, uint32
from numba.cuda.compiler import compile_ptx
from numba.cuda.testing import skip_on_cudasim, unittest
Just make sure we can compile this
        