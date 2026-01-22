import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
def test_entropy_transposed(self):
    pk = np.array([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
    qk = np.array([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
    assert_array_almost_equal(stats.entropy(pk.T, qk.T).T, stats.entropy(pk, qk, axis=1))