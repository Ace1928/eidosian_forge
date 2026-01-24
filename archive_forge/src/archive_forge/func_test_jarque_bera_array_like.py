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
def test_jarque_bera_array_like(self):
    np.random.seed(989284321)
    x = np.random.normal(0, 1, 100000)
    jb_test1 = JB1, p1 = stats.jarque_bera(list(x))
    jb_test2 = JB2, p2 = stats.jarque_bera(tuple(x))
    jb_test3 = JB3, p3 = stats.jarque_bera(x.reshape(2, 50000))
    assert JB1 == JB2 == JB3 == jb_test1.statistic == jb_test2.statistic == jb_test3.statistic
    assert p1 == p2 == p3 == jb_test1.pvalue == jb_test2.pvalue == jb_test3.pvalue