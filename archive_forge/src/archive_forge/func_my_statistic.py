import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def my_statistic(x, y, z, axis=-1):
    return x.mean(axis=axis) + y.mean(axis=axis) + z.mean(axis=axis)