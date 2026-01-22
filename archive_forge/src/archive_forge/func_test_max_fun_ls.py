from scipy.optimize import dual_annealing, Bounds
from scipy.optimize._dual_annealing import EnergyState
from scipy.optimize._dual_annealing import LocalSearchWrapper
from scipy.optimize._dual_annealing import ObjectiveFunWrapper
from scipy.optimize._dual_annealing import StrategyChain
from scipy.optimize._dual_annealing import VisitingDistribution
from scipy.optimize import rosen, rosen_der
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_array_less
from pytest import raises as assert_raises
from scipy._lib._util import check_random_state
def test_max_fun_ls(self):
    ret = dual_annealing(self.func, self.ld_bounds, maxfun=100, seed=self.seed)
    ls_max_iter = min(max(len(self.ld_bounds) * LocalSearchWrapper.LS_MAXITER_RATIO, LocalSearchWrapper.LS_MAXITER_MIN), LocalSearchWrapper.LS_MAXITER_MAX)
    assert ret.nfev <= 100 + ls_max_iter
    assert not ret.success