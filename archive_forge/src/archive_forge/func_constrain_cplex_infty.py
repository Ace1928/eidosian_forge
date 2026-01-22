import numpy as np
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.cplex_conif import (
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
def constrain_cplex_infty(v) -> None:
    """
    Limit values of vector v between +/- infinity as
    defined in the CPLEX package
    """
    import cplex as cpx
    n = len(v)
    for i in range(n):
        if v[i] >= cpx.infinity:
            v[i] = cpx.infinity
        if v[i] <= -cpx.infinity:
            v[i] = -cpx.infinity