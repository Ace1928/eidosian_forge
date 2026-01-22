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
def dijk(yi):
    n = len(yi)
    x = np.arange(n)
    dy = yi - yi[:, np.newaxis]
    dx = x - x[:, np.newaxis]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return dy[mask] / dx[mask]