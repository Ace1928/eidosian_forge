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
def test_agree_with_r(self):
    x = np.linspace(-1, 1, 9)
    self._testOne(x, 'two-sided', 0.15865525393145705, 0.9516406920151839)
    x = np.linspace(-15, 15, 9)
    self._testOne(x, 'two-sided', 0.4443560271592436, 0.038850140086788665)
    x = [-1.23, 0.06, -0.6, 0.17, 0.66, -0.17, -0.08, 0.27, -0.98, -0.99]
    self._testOne(x, 'two-sided', 0.293580126801961, 0.293408463684361)
    self._testOne(x, 'greater', 0.293580126801961, 0.146988835042376, mode='exact')
    self._testOne(x, 'less', 0.109348552425692, 0.732768892470675, mode='exact')