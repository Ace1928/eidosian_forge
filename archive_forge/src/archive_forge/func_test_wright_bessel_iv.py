import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
from scipy.special import rgamma, wright_bessel
@pytest.mark.parametrize('b', [0, 1e-06, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('x', [0, 1e-06, 0.1, 0.5, 1])
def test_wright_bessel_iv(b, x):
    """Test relation of wright_bessel and modified bessel function iv.

    iv(z) = (1/2*z)**v * Phi(1, v+1; 1/4*z**2).
    See https://dlmf.nist.gov/10.46.E2
    """
    if x != 0:
        v = b - 1
        wb = wright_bessel(1, v + 1, x ** 2 / 4.0)
        assert_allclose(np.power(x / 2.0, v) * wb, sc.iv(v, x), rtol=1e-11, atol=1e-11)