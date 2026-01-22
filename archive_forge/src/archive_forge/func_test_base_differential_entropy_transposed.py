import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
def test_base_differential_entropy_transposed(self):
    random_state = np.random.RandomState(0)
    values = random_state.standard_normal((3, 100))
    assert_allclose(stats.differential_entropy(values.T).T, stats.differential_entropy(values, axis=1))