import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_munp_expect(dist, args, msg):
    if dist._munp.__func__ != stats.rv_continuous._munp:
        res = dist.moment(5, *args)
        ref = dist.expect(lambda x: x ** 5, args, lb=-np.inf, ub=np.inf)
        if not np.isfinite(res):
            return
        assert_allclose(res, ref, atol=1e-10, rtol=0.0001, err_msg=msg + ' - higher moment / _munp')