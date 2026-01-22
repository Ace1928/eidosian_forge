import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
def test_accuracy_3(self):
    A, B, opt_perm = chr12c()
    res = quadratic_assignment(A, B, method=self.method, options={'rng': 0})
    assert_(11156 <= res.fun < 21000)
    assert_equal(res.fun, _score(A, B, res.col_ind))
    res = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'maximize': True})
    assert_(74000 <= res.fun < 85000)
    assert_equal(res.fun, _score(A, B, res.col_ind))
    seed_cost = np.array([4, 8, 10])
    seed = np.asarray([seed_cost, opt_perm[seed_cost]]).T
    res = quadratic_assignment(A, B, method=self.method, options={'partial_match': seed})
    assert_(11156 <= res.fun < 21000)
    assert_equal(res.col_ind[seed_cost], opt_perm[seed_cost])
    seed = np.asarray([np.arange(len(A)), opt_perm]).T
    res = quadratic_assignment(A, B, method=self.method, options={'partial_match': seed})
    assert_equal(res.col_ind, seed[:, 1].T)
    assert_equal(res.fun, 11156)
    assert_equal(res.nit, 0)
    empty = np.empty((0, 0))
    res = quadratic_assignment(empty, empty, method=self.method, options={'rng': 0})
    assert_equal(res.nit, 0)
    assert_equal(res.fun, 0)