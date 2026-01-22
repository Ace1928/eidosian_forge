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
def testLarge(self):
    n1, n2 = (10000, 110)
    lcm = n1 * 11.0
    delta = 1.0 / n1 / n2 / 2 / 2
    x = np.linspace(1, 200, n1) - delta
    y = np.linspace(2, 100, n2)
    self._testOne(x, y, 'two-sided', 55275.0 / lcm, 4.218847493575595e-15)
    self._testOne(x, y, 'greater', 561.0 / lcm, 0.9911545458204759)
    self._testOne(x, y, 'less', 55275.0 / lcm, 3.1317328311518713e-26)