from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types
def test_extension_type_as_arg(self):

    @cuda.jit
    def f(r, x):
        iv = Interval(x[0], x[1])
        r[0] = interval_width(iv)
    x = np.asarray((1.5, 2.5))
    r = np.zeros(1)
    f[1, 1](r, x)
    np.testing.assert_allclose(r[0], x[1] - x[0])