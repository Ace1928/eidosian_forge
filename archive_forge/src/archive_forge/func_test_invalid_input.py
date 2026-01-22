from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
def test_invalid_input(self):
    y = np.arange(5)
    msg = 'x and y must contain at least two observations.'
    with pytest.raises(ValueError, match=msg):
        cramervonmises_2samp([], y)
    with pytest.raises(ValueError, match=msg):
        cramervonmises_2samp(y, [1])
    msg = 'method must be either auto, exact or asymptotic'
    with pytest.raises(ValueError, match=msg):
        cramervonmises_2samp(y, y, 'xyz')