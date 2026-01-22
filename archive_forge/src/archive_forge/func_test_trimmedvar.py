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
def test_trimmedvar(self):
    rng = np.random.default_rng(3262323289434724460)
    data_orig = rng.random(size=20)
    data = np.sort(data_orig)
    data = ma.array(data, mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    assert_allclose(mstats.trimmed_var(data_orig, 0.1), data.var())