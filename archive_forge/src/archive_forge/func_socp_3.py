import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def socp_3(axis) -> SolverTestHelper:
    x = cp.Variable(shape=(2,))
    c = np.array([-1, 2])
    root2 = np.sqrt(2)
    u = np.array([[1 / root2, -1 / root2], [1 / root2, 1 / root2]])
    mat1 = np.diag([root2, 1 / root2]) @ u.T
    mat2 = np.diag([1, 1])
    mat3 = np.diag([0.2, 1.8])
    X = cp.vstack([mat1 @ x, mat2 @ x, mat3 @ x])
    t = cp.Constant(np.ones(3))
    objective = cp.Minimize(c @ x)
    if axis == 0:
        con = cp.constraints.SOC(t, X.T, axis=0)
        con_expect = [np.array([0, 1.16454469, 0.767560451]), np.array([[0, -0.974311819, -0.12844086], [0, 0.637872081, 0.756737724]])]
    else:
        con = cp.constraints.SOC(t, X, axis=1)
        con_expect = [np.array([0, 1.16454469, 0.767560451]), np.array([[0, 0], [-0.974311819, 0.637872081], [-0.12844086, 0.756737724]])]
    obj_pair = (objective, -1.932105)
    con_pairs = [(con, con_expect)]
    var_pairs = [(x, np.array([0.83666003, -0.54772256]))]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth