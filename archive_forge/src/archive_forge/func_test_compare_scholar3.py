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
def test_compare_scholar3(self):
    """
        Data taken from 'Robustness And Comparative Power Of WelchAspin,
        Alexander-Govern And Yuen Tests Under Non-Normality And Variance
        Heteroscedasticity', by Ayed A. Almoied. 2017. Page 34-37.
        https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=2775&context=oa_dissertations
        """
    x1 = [-1.77559, -1.4113, -0.69457, -0.54148, -0.18808, -0.07152, 0.04696, 0.051183, 0.148695, 0.168052, 0.422561, 0.458555, 0.616123, 0.709968, 0.839956, 0.857226, 0.929159, 0.981442, 0.999554, 1.642958]
    x2 = [-1.47973, -1.2722, -0.91914, -0.80916, -0.75977, -0.72253, -0.3601, -0.33273, -0.28859, -0.09637, -0.08969, -0.01824, 0.260131, 0.289278, 0.518254, 0.683003, 0.877618, 1.172475, 1.33964, 1.576766]
    soln = stats.alexandergovern(x1, x2)
    assert_allclose(soln.statistic, 0.713526, atol=1e-05)
    assert_allclose(soln.pvalue, 0.398276, atol=1e-05)
    '\n        tested in ag.test in R:\n        > library("onewaytests")\n        > library("tibble")\n        > x1 <- c(-1.77559, -1.4113, -0.69457, -0.54148, -0.18808, -0.07152,\n        +          0.04696, 0.051183, 0.148695, 0.168052, 0.422561, 0.458555,\n        +          0.616123, 0.709968, 0.839956, 0.857226, 0.929159, 0.981442,\n        +          0.999554, 1.642958)\n        > x2 <- c(-1.47973, -1.2722, -0.91914, -0.80916, -0.75977, -0.72253,\n        +         -0.3601, -0.33273, -0.28859, -0.09637, -0.08969, -0.01824,\n        +         0.260131, 0.289278, 0.518254, 0.683003, 0.877618, 1.172475,\n        +         1.33964, 1.576766)\n        > x1_fact <- c(rep("x1", times=20))\n        > x2_fact <- c(rep("x2", times=20))\n        > a <- c(x1, x2)\n        > b <- factor(c(x1_fact, x2_fact))\n        > ag.test(a ~ b, tibble(a, b))\n        Alexander-Govern Test (alpha = 0.05)\n        -------------------------------------------------------------\n        data : a and b\n\n        statistic  : 0.7135182\n        parameter  : 1\n        p.value    : 0.3982783\n\n        Result     : Difference is not statistically significant.\n        -------------------------------------------------------------\n        '
    assert_allclose(soln.statistic, 0.7135182)
    assert_allclose(soln.pvalue, 0.3982783)