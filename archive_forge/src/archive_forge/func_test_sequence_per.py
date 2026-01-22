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
def test_sequence_per(self):
    x = arange(8) * 0.5
    expected = np.array([0, 3.5, 1.75])
    res = stats.scoreatpercentile(x, [0, 100, 50])
    assert_allclose(res, expected)
    assert_(isinstance(res, np.ndarray))
    assert_allclose(stats.scoreatpercentile(x, np.array([0, 100, 50])), expected)
    res2 = stats.scoreatpercentile(np.arange(12).reshape((3, 4)), np.array([0, 1, 100, 100]), axis=1)
    expected2 = array([[0, 4, 8], [0.03, 4.03, 8.03], [3, 7, 11], [3, 7, 11]])
    assert_allclose(res2, expected2)