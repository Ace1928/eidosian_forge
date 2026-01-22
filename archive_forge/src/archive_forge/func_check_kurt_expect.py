import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_kurt_expect(distfn, arg, m, v, k, msg):
    if np.isfinite(k):
        m4e = distfn.expect(lambda x: np.power(x - m, 4), arg)
        npt.assert_allclose(m4e, (k + 3.0) * np.power(v, 2), atol=1e-05, rtol=1e-05, err_msg=msg + ' - kurtosis')
    elif not np.isposinf(k):
        npt.assert_(np.isnan(k))