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
def test_zscore_constant_input_2d_nan_policy_omit(self):
    x = np.array([[10.0, 10.0, 10.0, 10.0], [10.0, 11.0, 12.0, np.nan], [10.0, 12.0, np.nan, 10.0]])
    z0 = stats.zscore(x, nan_policy='omit', axis=0)
    s = np.sqrt(3 / 2)
    s2 = np.sqrt(2)
    assert_allclose(z0, np.array([[np.nan, -s, -1.0, np.nan], [np.nan, 0, 1.0, np.nan], [np.nan, s, np.nan, np.nan]]))
    z1 = stats.zscore(x, nan_policy='omit', axis=1)
    assert_allclose(z1, np.array([[np.nan, np.nan, np.nan, np.nan], [-s, 0, s, np.nan], [-s2 / 2, s2, np.nan, -s2 / 2]]))