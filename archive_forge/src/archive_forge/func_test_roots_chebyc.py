import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def test_roots_chebyc():
    weightf = orth.chebyc(5).weight_func
    verify_gauss_quad(sc.roots_chebyc, sc.eval_chebyc, weightf, -2.0, 2.0, 5)
    verify_gauss_quad(sc.roots_chebyc, sc.eval_chebyc, weightf, -2.0, 2.0, 25)
    verify_gauss_quad(sc.roots_chebyc, sc.eval_chebyc, weightf, -2.0, 2.0, 100, atol=1e-12)
    x, w = sc.roots_chebyc(5, False)
    y, v, m = sc.roots_chebyc(5, True)
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)
    muI, muI_err = integrate.quad(weightf, -2, 2)
    assert_allclose(m, muI, rtol=muI_err)
    assert_raises(ValueError, sc.roots_chebyc, 0)
    assert_raises(ValueError, sc.roots_chebyc, 3.3)