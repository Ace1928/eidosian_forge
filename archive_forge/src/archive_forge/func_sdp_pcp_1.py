import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def sdp_pcp_1() -> SolverTestHelper:
    """
    Example sdp and power cone.
    """
    Sigma = np.array([[0.4787481, -0.96924914], [-0.96924914, 2.77788598]])
    x = cp.Variable(shape=(2, 1))
    y = cp.Variable(shape=(2, 1))
    X = cp.Variable(shape=(2, 2), symmetric=True)
    M1 = cp.vstack([X, x.T])
    M2 = cp.vstack([x, np.ones((1, 1))])
    M3 = cp.hstack([M1, M2])
    var_pairs = [(x, np.array([[0.72128204], [0.27871796]])), (y, np.array([[0.01], [0.01]])), (X, np.array([[0.52024779, 0.20103426], [0.20103426, 0.0776837]]))]
    con_pairs = [(cp.sum(x) == 1, -0.1503204799112807), (x >= 0, np.array([[-0.0], [-0.0]])), (y >= 0.01, np.array([[0.70705506], [0.70715844]])), (M3 >> 0, np.array([[0.4787481, -0.96924914, -0.07516024], [-0.96924914, 2.77788598, -0.07516024], [-0.07516024, -0.07516024, 0.07516094]])), (cp.PowCone3D(x, np.ones((2, 1)), y, 0.9), [np.array([[1.17878172e-09], [3.05162243e-09]]), np.array([[9.2015764e-10], [9.40823207e-10]]), np.array([[2.41053358e-10], [7.43432462e-10]])])]
    obj_expr = cp.Minimize(cp.trace(Sigma @ X) + cp.norm(y, p=2))
    obj_pair = (obj_expr, 0.089301671322676)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth