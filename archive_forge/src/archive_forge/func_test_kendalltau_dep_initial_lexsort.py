import os
import re
import warnings
from collections import namedtuple
from itertools import product
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
from numpy.lib import NumpyVersion
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
def test_kendalltau_dep_initial_lexsort():
    with pytest.warns(DeprecationWarning, match="'kendalltau' keyword argument 'initial_lexsort'"):
        stats.kendalltau([], [], initial_lexsort=True)