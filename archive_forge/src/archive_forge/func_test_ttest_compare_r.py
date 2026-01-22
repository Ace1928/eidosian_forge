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
@pytest.mark.parametrize('a,b,pr,tr,trim', params)
def test_ttest_compare_r(self, a, b, pr, tr, trim):
    """
        Using PairedData's yuen.t.test method. Something to note is that there
        are at least 3 R packages that come with a trimmed t-test method, and
        comparisons were made between them. It was found that PairedData's
        method's results match this method, SAS, and one of the other R
        methods. A notable discrepancy was the DescTools implementation of the
        function, which only sometimes agreed with SAS, WRS2, PairedData and
        this implementation. For this reason, most comparisons in R are made
        against PairedData's method.

        Rather than providing the input and output for all evaluations, here is
        a representative example:
        > library(PairedData)
        > a <- c(1, 2, 3)
        > b <- c(1.1, 2.9, 4.2)
        > options(digits=16)
        > yuen.t.test(a, b, tr=.2)

            Two-sample Yuen test, trim=0.2

        data:  x and y
        t = -0.68649512735573, df = 3.4104431643464, p-value = 0.5361949075313
        alternative hypothesis: true difference in trimmed means is not equal
        to 0
        95 percent confidence interval:
         -3.912777195645217  2.446110528978550
        sample estimates:
        trimmed mean of x trimmed mean of y
        2.000000000000000 2.73333333333333
        """
    statistic, pvalue = stats.ttest_ind(a, b, trim=trim, equal_var=False)
    assert_allclose(statistic, tr, atol=1e-15)
    assert_allclose(pvalue, pr, atol=1e-15)