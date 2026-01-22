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
def test_binomtest():
    pp = np.concatenate((np.linspace(0.1, 0.2, 5), np.linspace(0.45, 0.65, 5), np.linspace(0.85, 0.95, 5)))
    n = 501
    x = 450
    results = [0.0, 0.0, 1.0159969301994141e-304, 2.975241857215053e-275, 7.766838292253527e-250, 2.3381250925167094e-99, 7.828459158732395e-81, 9.915594781996138e-65, 2.872939072517631e-50, 1.717506629838842e-37, 0.0021070691951093692, 0.12044570587262322, 0.8815476317480251, 0.027120993063129286, 2.610258713469472e-06]
    for p, res in zip(pp, results):
        assert_approx_equal(stats.binomtest(x, n, p).pvalue, res, significant=12, err_msg='fail forp=%f' % p)
    assert_approx_equal(stats.binomtest(50, 100, 0.1).pvalue, 5.832038785734365e-24, significant=12)