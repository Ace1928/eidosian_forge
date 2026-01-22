import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.special import log_ndtr, ndtri_exp
from scipy.special._testutils import assert_func_equal
@pytest.fixture(scope='class')
def uniform_random_points():
    random_state = np.random.RandomState(1234)
    points = random_state.random_sample(1000)
    return points