import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
@property
def constr(self):

    def fun(x):
        x_coord, y_coord, z_coord = self._get_cordinates(x)
        return x_coord ** 2 + y_coord ** 2 + z_coord ** 2 - 1
    if self.constr_jac is None:

        def jac(x):
            x_coord, y_coord, z_coord = self._get_cordinates(x)
            Jx = 2 * np.diag(x_coord)
            Jy = 2 * np.diag(y_coord)
            Jz = 2 * np.diag(z_coord)
            return csc_matrix(np.hstack((Jx, Jy, Jz)))
    else:
        jac = self.constr_jac
    if self.constr_hess is None:

        def hess(x, v):
            D = 2 * np.diag(v)
            return block_diag(D, D, D)
    else:
        hess = self.constr_hess
    return NonlinearConstraint(fun, -np.inf, 0, jac, hess)