import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def test_roots_hermite():
    rootf = sc.roots_hermite
    evalf = sc.eval_hermite
    weightf = orth.hermite(5).weight_func
    verify_gauss_quad(rootf, evalf, weightf, -np.inf, np.inf, 5)
    verify_gauss_quad(rootf, evalf, weightf, -np.inf, np.inf, 25, atol=1e-13)
    verify_gauss_quad(rootf, evalf, weightf, -np.inf, np.inf, 100, atol=1e-12)
    x, w = sc.roots_hermite(5, False)
    y, v, m = sc.roots_hermite(5, True)
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)
    muI, muI_err = integrate.quad(weightf, -np.inf, np.inf)
    assert_allclose(m, muI, rtol=muI_err)
    x, w = sc.roots_hermite(200, False)
    y, v, m = sc.roots_hermite(200, True)
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)
    assert_allclose(sum(v), m, 1e-14, 1e-14)
    assert_raises(ValueError, sc.roots_hermite, 0)
    assert_raises(ValueError, sc.roots_hermite, 3.3)