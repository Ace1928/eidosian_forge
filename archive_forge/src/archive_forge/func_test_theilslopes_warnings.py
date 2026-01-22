import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
def test_theilslopes_warnings():
    msg = 'All `x` coordinates.*|Mean of empty slice.|invalid value encountered.*'
    with pytest.warns(RuntimeWarning, match=msg):
        res = mstats.theilslopes([0, 1], [0, 0])
        assert np.all(np.isnan(res))
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered...')
        res = mstats.theilslopes([0, 0, 0], [0, 1, 0])
        assert_allclose(res, (0, 0, np.nan, np.nan))