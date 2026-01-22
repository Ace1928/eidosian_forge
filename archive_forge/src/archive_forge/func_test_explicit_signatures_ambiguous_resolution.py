import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@skip_on_cudasim('No overload resolution in the simulator')
def test_explicit_signatures_ambiguous_resolution(self):
    f = cuda.jit(['(float64[::1], float32, float64)', '(float64[::1], float64, float32)', '(float64[::1], int64, int64)'])(add_kernel)
    with self.assertRaises(TypeError) as cm:
        r = np.zeros(1, dtype=np.float64)
        f[1, 1](r, 1.0, 2.0)
    self.assertRegex(str(cm.exception), "Ambiguous overloading for <function add_kernel [^>]*> \\(Array\\(float64, 1, 'C', False, aligned=True\\), float64, float64\\):\\n\\(Array\\(float64, 1, 'C', False, aligned=True\\), float32, float64\\) -> none\\n\\(Array\\(float64, 1, 'C', False, aligned=True\\), float64, float32\\) -> none")
    self.assertNotIn('int64', str(cm.exception))