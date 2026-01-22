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
def test_binomtest2():
    res2 = [[1.0, 1.0], [0.5, 1.0, 0.5], [0.25, 1.0, 1.0, 0.25], [0.125, 0.625, 1.0, 0.625, 0.125], [0.0625, 0.375, 1.0, 1.0, 0.375, 0.0625], [0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875, 0.03125], [0.015625, 0.125, 0.453125, 1.0, 1.0, 0.453125, 0.125, 0.015625], [0.0078125, 0.0703125, 0.2890625, 0.7265625, 1.0, 0.7265625, 0.2890625, 0.0703125, 0.0078125], [0.00390625, 0.0390625, 0.1796875, 0.5078125, 1.0, 1.0, 0.5078125, 0.1796875, 0.0390625, 0.00390625], [0.001953125, 0.021484375, 0.109375, 0.34375, 0.75390625, 1.0, 0.75390625, 0.34375, 0.109375, 0.021484375, 0.001953125]]
    for k in range(1, 11):
        res1 = [stats.binomtest(v, k, 0.5).pvalue for v in range(k + 1)]
        assert_almost_equal(res1, res2[k - 1], decimal=10)