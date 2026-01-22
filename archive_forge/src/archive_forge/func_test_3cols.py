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
def test_3cols(self):
    x1 = np.arange(6)
    x2 = -x1
    x3 = np.array([0, 1, 2, 3, 5, 4])
    x = np.asarray([x1, x2, x3]).T
    actual = stats.spearmanr(x)
    expected_corr = np.array([[1, -1, 0.94285714], [-1, 1, -0.94285714], [0.94285714, -0.94285714, 1]])
    expected_pvalue = np.zeros((3, 3), dtype=float)
    expected_pvalue[2, 0:2] = 0.00480466472
    expected_pvalue[0:2, 2] = 0.00480466472
    assert_allclose(actual.statistic, expected_corr)
    assert_allclose(actual.pvalue, expected_pvalue)