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
@pytest.mark.parametrize(('axis', 'fun_name', 'p'), [(None, 'wpmean_reference', 9.87654321), (0, 'gmean', 0), (1, 'hmean', -1)])
def test_weights_2d_array(self, axis, fun_name, p):
    if fun_name == 'wpmean_reference':

        def fun(a, axis, weights):
            return TestPowMean.wpmean_reference(a, p, weights)
    else:
        fun = getattr(stats, fun_name)
    a = np.array([[2, 5], [10, 5], [6, 5]])
    weights = np.array([[10, 1], [5, 1], [3, 1]])
    desired = fun(a, axis=axis, weights=weights)
    check_equal_pmean(a, p, desired, axis=axis, weights=weights, rtol=1e-05)