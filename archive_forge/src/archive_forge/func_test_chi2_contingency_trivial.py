import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.special import xlogy
from scipy.stats.contingency import (margins, expected_freq,
def test_chi2_contingency_trivial():
    obs = np.array([[1, 2], [1, 2]])
    chi2, p, dof, expected = chi2_contingency(obs, correction=False)
    assert_equal(chi2, 0.0)
    assert_equal(p, 1.0)
    assert_equal(dof, 1)
    assert_array_equal(obs, expected)
    obs = np.array([1, 2, 3])
    chi2, p, dof, expected = chi2_contingency(obs, correction=False)
    assert_equal(chi2, 0.0)
    assert_equal(p, 1.0)
    assert_equal(dof, 0)
    assert_array_equal(obs, expected)