import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from multiprocessing.dummy import Pool
def test_quad_vec_pool():
    f = _lorenzian
    res, err = quad_vec(f, -np.inf, np.inf, norm='max', epsabs=0.0001, workers=4)
    assert_allclose(res, np.pi, rtol=0, atol=0.0001)
    with Pool(10) as pool:

        def f(x):
            return 1 / (1 + x ** 2)
        res, _ = quad_vec(f, -np.inf, np.inf, norm='max', epsabs=0.0001, workers=pool.map)
        assert_allclose(res, np.pi, rtol=0, atol=0.0001)