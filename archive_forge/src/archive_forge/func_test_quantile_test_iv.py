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
def test_quantile_test_iv(self):
    x = [1, 2, 3]
    message = '`x` must be a one-dimensional array of numbers.'
    with pytest.raises(ValueError, match=message):
        stats.quantile_test([x])
    message = '`q` must be a scalar.'
    with pytest.raises(ValueError, match=message):
        stats.quantile_test(x, q=[1, 2])
    message = '`p` must be a float strictly between 0 and 1.'
    with pytest.raises(ValueError, match=message):
        stats.quantile_test(x, p=[0.5, 0.75])
    with pytest.raises(ValueError, match=message):
        stats.quantile_test(x, p=2)
    with pytest.raises(ValueError, match=message):
        stats.quantile_test(x, p=-0.5)
    message = '`alternative` must be one of...'
    with pytest.raises(ValueError, match=message):
        stats.quantile_test(x, alternative='one-sided')
    message = '`confidence_level` must be a number between 0 and 1.'
    with pytest.raises(ValueError, match=message):
        stats.quantile_test(x).confidence_interval(1)