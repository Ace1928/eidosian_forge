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
def test_regress_against_R(self):
    x = [151, 174, 138, 186, 128, 136, 179, 163, 152, 131]
    y = [63, 81, 56, 91, 47, 57, 76, 72, 62, 48]
    res = stats.linregress(x, y, alternative='two-sided')
    assert_allclose(res.slope, 0.6746104491292)
    assert_allclose(res.intercept, -38.455087076077)
    assert_allclose(res.rvalue, np.sqrt(0.95478224775))
    assert_allclose(res.pvalue, 1.16440531074e-06)
    assert_allclose(res.stderr, 0.0519051424731)
    assert_allclose(res.intercept_stderr, 8.0490133029927)