import numpy as np
import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
def makeMstart(A, n, ifCol: int=1):
    mstart = np.bincount(A.nonzero()[ifCol])
    mstart = np.concatenate((np.array([0], dtype=np.int64), mstart, np.array([0] * (n - len(mstart)), dtype=np.int64)))
    mstart = np.cumsum(mstart)
    return mstart