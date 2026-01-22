import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
from scipy.special import rgamma, wright_bessel
@pytest.mark.parametrize('a', [0, 1e-06, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('b', [0, 1e-06, 0.1, 0.5, 1, 10])
def test_wright_bessel_zero(a, b):
    """Test at x = 0."""
    assert_equal(wright_bessel(a, b, 0.0), rgamma(b))