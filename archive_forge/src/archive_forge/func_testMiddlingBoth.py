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
@pytest.mark.slow
def testMiddlingBoth(self):
    n1, n2 = (500, 600)
    delta = 1.0 / n1 / n2 / 2 / 2
    x = np.linspace(1, 200, n1) - delta
    y = np.linspace(2, 200, n2)
    self._testOne(x, y, 'two-sided', 2000.0 / n1 / n2, 1.0, mode='auto')
    self._testOne(x, y, 'two-sided', 2000.0 / n1 / n2, 1.0, mode='asymp')
    self._testOne(x, y, 'greater', 2000.0 / n1 / n2, 0.9697596024683929, mode='asymp')
    self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021, mode='asymp')
    with suppress_warnings() as sup:
        message = 'ks_2samp: Exact calculation unsuccessful.'
        sup.filter(RuntimeWarning, message)
        self._testOne(x, y, 'greater', 2000.0 / n1 / n2, 0.9697596024683929, mode='exact')
        self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021, mode='exact')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021, mode='exact')
        _check_warnings(w, RuntimeWarning, 1)