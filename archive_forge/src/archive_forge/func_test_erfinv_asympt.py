import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
import scipy.special as sc
def test_erfinv_asympt(self):
    x = np.array([1e-20, 1e-15, 1e-14, 1e-10, 1e-08, 9e-08, 1.1e-07, 1e-06])
    expected = np.array([8.86226925452758e-21, 8.862269254527581e-16, 8.86226925452758e-15, 8.862269254527581e-11, 8.86226925452758e-09, 7.97604232907484e-08, 9.74849617998037e-08, 8.8622692545299e-07])
    assert_allclose(sc.erfinv(x), expected, rtol=1e-15)
    assert_allclose(sc.erf(sc.erfinv(x)), x, rtol=5e-15)