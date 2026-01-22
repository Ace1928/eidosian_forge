import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def pcp_2() -> SolverTestHelper:
    """
    Reformulate

        max  (x**0.2)*(y**0.8) + z**0.4 - x
        s.t. x + y + z/2 == 2
             x, y, z >= 0
    Into

        max  x3 + x4 - x0
        s.t. x0 + x1 + x2 / 2 == 2,
             (x0, x1, x3) in Pow3D(0.2)
             (x2, 1.0, x4) in Pow3D(0.4)
    """
    x = cp.Variable(shape=(3,))
    hypos = cp.Variable(shape=(2,))
    objective = cp.Minimize(-cp.sum(hypos) + x[0])
    arg1 = cp.hstack([x[0], x[2]])
    arg2 = cp.hstack([x[1], 1.0])
    pc_con = cp.constraints.PowCone3D(arg1, arg2, hypos, [0.2, 0.4])
    expect_pc_con = [np.array([1.48466366, 0.24233184]), np.array([0.48466367, 0.83801333]), np.array([-1.0, -1.0])]
    con_pairs = [(x[0] + x[1] + 0.5 * x[2] == 2, 0.4846636697795672), (pc_con, expect_pc_con)]
    obj_pair = (objective, -1.8073406786220672)
    var_pairs = [(x, np.array([0.06393515, 0.78320961, 2.30571048])), (hypos, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth