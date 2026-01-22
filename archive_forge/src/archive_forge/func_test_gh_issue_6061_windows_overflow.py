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
def test_gh_issue_6061_windows_overflow(self):
    x = list(range(2000))
    y = list(range(2000))
    y[0], y[9] = (y[9], y[0])
    y[10], y[434] = (y[434], y[10])
    y[435], y[1509] = (y[1509], y[435])
    x.append(np.nan)
    y.append(3.0)
    assert_almost_equal(stats.spearmanr(x, y, nan_policy='omit')[0], 0.998)