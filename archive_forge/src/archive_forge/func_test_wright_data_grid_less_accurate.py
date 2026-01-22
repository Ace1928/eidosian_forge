import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
from scipy.special import rgamma, wright_bessel
@pytest.mark.parametrize('a, b, x, phi, accuracy', grid_a_b_x_value_acc.tolist())
def test_wright_data_grid_less_accurate(a, b, x, phi, accuracy):
    """Test cases of test_data that do not reach relative accuracy of 1e-11

    Here we test for reduced accuracy or even nan.
    """
    if np.isnan(accuracy):
        assert np.isnan(wright_bessel(a, b, x))
    else:
        assert_allclose(wright_bessel(a, b, x), phi, rtol=accuracy)