import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.special import stdtr, stdtrit, ndtr, ndtri
def test_stdtr_stdtri_invalid():
    df = [10000000000.0, 1000000000000.0, 1e+120, np.inf]
    x = np.nan
    res1 = stdtr(df, x)
    res2 = stdtrit(df, x)
    res_ex = 4 * [np.nan]
    assert_equal(res1, res_ex)
    assert_equal(res2, res_ex)