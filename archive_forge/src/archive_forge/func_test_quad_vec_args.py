import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from multiprocessing.dummy import Pool
def test_quad_vec_args():

    def f(x, a):
        return x * (x + a) * np.arange(3)
    a = 2
    exact = np.array([0, 4 / 3, 8 / 3])
    res, err = quad_vec(f, 0, 1, args=(a,))
    assert_allclose(res, exact, rtol=0, atol=0.0001)