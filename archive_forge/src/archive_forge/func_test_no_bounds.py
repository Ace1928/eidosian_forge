import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_no_bounds(self):
    x0 = np.zeros(3)
    h = np.full(3, 0.01)
    inf_lower = np.empty_like(x0)
    inf_upper = np.empty_like(x0)
    inf_lower.fill(-np.inf)
    inf_upper.fill(np.inf)
    h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', inf_lower, inf_upper)
    assert_allclose(h_adjusted, h)
    assert_(np.all(one_sided))
    h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', inf_lower, inf_upper)
    assert_allclose(h_adjusted, h)
    assert_(np.all(one_sided))
    h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 1, '2-sided', inf_lower, inf_upper)
    assert_allclose(h_adjusted, h)
    assert_(np.all(~one_sided))
    h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 2, '2-sided', inf_lower, inf_upper)
    assert_allclose(h_adjusted, h)
    assert_(np.all(~one_sided))