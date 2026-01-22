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
def test_known_examples(self):
    x = stats.norm.rvs(loc=0.2, size=100, random_state=987654321)
    self._testOne(x, 'two-sided', 0.12464329735846891, 0.08944488871182077, mode='asymp')
    self._testOne(x, 'less', 0.12464329735846891, 0.04098916407764175)
    self._testOne(x, 'greater', 0.007211523321631099, 0.9853115859039623)