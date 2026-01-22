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
def test_kendalltau_gh18139_overflow():
    import random
    random.seed(6272161)
    classes = [1, 2, 3, 4, 5, 6, 7]
    n_samples = 2 * 10 ** 5
    x = random.choices(classes, k=n_samples)
    y = random.choices(classes, k=n_samples)
    res = stats.kendalltau(x, y)
    assert_allclose(res.statistic, 0.0011816493905730343)
    assert_allclose(res.pvalue, 0.4894, atol=0.002)