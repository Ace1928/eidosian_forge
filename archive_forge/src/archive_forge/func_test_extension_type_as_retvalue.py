from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types
def test_extension_type_as_retvalue(self):

    @cuda.jit
    def f(r, x):
        iv1 = Interval(x[0], x[1])
        iv2 = Interval(x[2], x[3])
        iv_sum = sum_intervals(iv1, iv2)
        r[0] = iv_sum.lo
        r[1] = iv_sum.hi
    x = np.asarray((1.5, 2.5, 3.0, 4.0))
    r = np.zeros(2)
    f[1, 1](r, x)
    expected = np.asarray((x[0] + x[2], x[1] + x[3]))
    np.testing.assert_allclose(r, expected)