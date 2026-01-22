import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import pytest
import statsmodels.genmod.families as families
from statsmodels.tools import numdiff as nd
def test_inverse_deriv2():
    np.random.seed(24235)
    for link in LinksISD:
        for k in range(10):
            z = get_domainvalue(link)
            d2 = link.inverse_deriv2(z)
            d2a = nd.approx_fprime(np.r_[z], link.inverse_deriv)
            assert_allclose(d2, d2a, rtol=5e-06, atol=1e-06, err_msg=str(link))