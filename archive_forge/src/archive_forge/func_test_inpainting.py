from __future__ import print_function
import unittest
import numpy as np
import cvxpy as cvx
import cvxpy.interface as intf
from cvxpy.reductions.solvers.conic_solvers import ecos_conif
from cvxpy.tests.base_test import BaseTest
def test_inpainting(self) -> None:
    """Test image in-painting.
        """
    import numpy as np
    np.random.seed(1)
    rows, cols = (20, 20)
    Uorig = np.random.randint(0, 255, size=(rows, cols))
    rows, cols = Uorig.shape
    Known = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.random.random() > 0.7:
                Known[i, j] = 1
    Ucorr = Known * Uorig
    U = cvx.Variable((rows, cols))
    obj = cvx.Minimize(cvx.tv(U))
    constraints = [cvx.multiply(Known, U) == cvx.multiply(Known, Ucorr)]
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.SCS)