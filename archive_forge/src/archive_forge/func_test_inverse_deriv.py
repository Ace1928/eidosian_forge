import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import pytest
import statsmodels.genmod.families as families
from statsmodels.tools import numdiff as nd
def test_inverse_deriv():
    np.random.seed(24235)
    for link in Links:
        for k in range(10):
            z = get_domainvalue(link)
            d = link.inverse_deriv(z)
            f = 1 / link.deriv(link.inverse(z))
            assert_allclose(d, f, rtol=1e-08, atol=1e-10, err_msg=str(link))