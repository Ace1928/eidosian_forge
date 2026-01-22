import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
def test_log_ndtr_values_8_16(self):
    x = np.array([8.001, 8.06, 8.15, 8.5, 10, 12, 14, 16])
    expected = [-6.170639424817055e-16, -3.814722443652823e-16, -1.819621363526629e-16, -9.479534822203318e-18, -7.619853024160525e-24, -1.776482112077679e-33, -7.7935368191928e-45, -6.388754400538087e-58]
    y = sc.log_ndtr(x)
    assert_allclose(y, expected, rtol=5e-14)