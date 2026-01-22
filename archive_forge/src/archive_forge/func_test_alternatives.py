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
@pytest.mark.parametrize('alt,pr,tr', (('greater', 0.9985605452443, -4.2461168970325), ('less', 0.001439454755672, -4.2461168970325)))
def test_alternatives(self, alt, pr, tr):
    """
        > library(PairedData)
        > a <- c(2.7,2.7,1.1,3.0,1.9,3.0,3.8,3.8,0.3,1.9,1.9)
        > b <- c(6.5,5.4,8.1,3.5,0.5,3.8,6.8,4.9,9.5,6.2,4.1)
        > options(digits=16)
        > yuen.t.test(a, b, alternative = 'greater')
        """
    a = [2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9]
    b = [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1]
    statistic, pvalue = stats.ttest_ind(a, b, trim=0.2, equal_var=False, alternative=alt)
    assert_allclose(pvalue, pr, atol=1e-10)
    assert_allclose(statistic, tr, atol=1e-10)