import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_accept_gh7799(self):
    met = Metropolis(0)
    res_new = OptimizeResult(success=True, fun=0.0)
    res_old = OptimizeResult(success=True, fun=1.0)
    assert met(res_new=res_new, res_old=res_old)
    res_new.success = False
    assert not met(res_new=res_new, res_old=res_old)
    res_old.success = False
    assert met(res_new=res_new, res_old=res_old)