import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def mi_lp_6() -> SolverTestHelper:
    """Test MILP for timelimit and no feasible solution"""
    n = 70
    m = 70
    x = cp.Variable((n,), boolean=True, name='x')
    y = cp.Variable((n,), name='y')
    z = cp.Variable((m,), pos=True, name='z')
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    objective = cp.Maximize(cp.sum(y))
    constraints = [A @ y <= b, y <= 1, cp.sum(x) >= 10, cp.sum(x) <= 20, z[0] + z[1] + z[2] >= 10, z[3] + z[4] + z[5] >= 5, z[6] + z[7] + z[8] >= 7, z[9] + z[10] >= 8, z[11] + z[12] >= 6, z[13] + z[14] >= 3, z[15] + z[16] >= 2, z[17] + z[18] >= 1, z[19] >= 2, z[20] >= 1, z[21] >= 1, z[22] >= 1, z[23] >= 1, z[24] >= 1, z[25] >= 1, z[26] >= 1, z[27] >= 1, z[28] >= 1, z[29] >= 1]
    return SolverTestHelper((objective, None), [(x, None), (y, None), (z, None)], [(con, None) for con in constraints])