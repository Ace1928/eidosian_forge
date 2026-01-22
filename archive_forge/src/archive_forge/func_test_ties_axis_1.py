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
def test_ties_axis_1(self):
    z1 = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
    z2 = np.array([[1, 2, 3, 4], [1, 1, 1, 1]])
    z3 = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    warn_msg = 'An input array is constant'
    with assert_warns(stats.ConstantInputWarning, match=warn_msg):
        r, p = stats.spearmanr(z1, axis=1)
        assert_equal(r, np.nan)
        assert_equal(p, np.nan)
        r, p = stats.spearmanr(z2, axis=1)
        assert_equal(r, np.nan)
        assert_equal(p, np.nan)
        r, p = stats.spearmanr(z3, axis=1)
        assert_equal(r, np.nan)
        assert_equal(p, np.nan)