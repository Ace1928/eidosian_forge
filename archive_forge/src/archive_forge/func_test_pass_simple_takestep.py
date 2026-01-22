import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_pass_simple_takestep(self):
    takestep = myTakeStep2
    i = 1
    res = basinhopping(func2d_nograd, self.x0[i], minimizer_kwargs=self.kwargs_nograd, niter=self.niter, disp=self.disp, take_step=takestep)
    assert_almost_equal(res.x, self.sol[i], self.tol)