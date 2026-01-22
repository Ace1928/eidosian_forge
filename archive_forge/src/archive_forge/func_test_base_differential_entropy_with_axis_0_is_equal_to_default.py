import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
def test_base_differential_entropy_with_axis_0_is_equal_to_default(self):
    random_state = np.random.RandomState(0)
    values = random_state.standard_normal((100, 3))
    entropy = stats.differential_entropy(values, axis=0)
    default_entropy = stats.differential_entropy(values)
    assert_allclose(entropy, default_entropy)