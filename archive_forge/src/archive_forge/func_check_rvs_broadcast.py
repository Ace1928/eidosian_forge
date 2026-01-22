import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_rvs_broadcast(distfunc, distname, allargs, shape, shape_only, otype):
    np.random.seed(123)
    sample = distfunc.rvs(*allargs)
    assert_equal(sample.shape, shape, '%s: rvs failed to broadcast' % distname)
    if not shape_only:
        rvs = np.vectorize(lambda *allargs: distfunc.rvs(*allargs), otypes=otype)
        np.random.seed(123)
        expected = rvs(*allargs)
        assert_allclose(sample, expected, rtol=1e-13)