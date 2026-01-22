from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector
def test_args_kwargs(self):
    a = 3.0
    for jac in ['2-point', '3-point', 'cs', jac_trivial]:
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
            res = least_squares(fun_trivial, 2.0, jac, args=(a,), method=self.method)
            res1 = least_squares(fun_trivial, 2.0, jac, kwargs={'a': a}, method=self.method)
        assert_allclose(res.x, a, rtol=0.0001)
        assert_allclose(res1.x, a, rtol=0.0001)
        assert_raises(TypeError, least_squares, fun_trivial, 2.0, args=(3, 4), method=self.method)
        assert_raises(TypeError, least_squares, fun_trivial, 2.0, kwargs={'kaboom': 3}, method=self.method)