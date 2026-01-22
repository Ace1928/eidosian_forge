import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
@pytest.mark.parametrize('nan_policy, expected', [('omit', np.array([np.nan, 1.5, 1.5])), ('propagate', np.array([np.nan, np.nan, 1.5]))])
def test_nan_policy_with_axis(self, nan_policy, expected):
    x = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [1, 5, 3, 6, np.nan, np.nan], [5, 6, 7, 9, 9, 10]])
    mad = stats.median_abs_deviation(x, nan_policy=nan_policy, axis=1)
    assert_equal(mad, expected)