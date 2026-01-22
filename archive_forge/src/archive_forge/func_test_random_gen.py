import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_random_gen(self):
    rng = np.random.default_rng(1)
    minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': True}
    res1 = basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, seed=rng)
    rng = np.random.default_rng(1)
    res2 = basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs, niter=10, seed=rng)
    assert_equal(res1.x, res2.x)