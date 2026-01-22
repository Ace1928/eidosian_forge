import numpy as np
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
def constrain_gurobi_infty(v) -> None:
    """
    Limit values of vector v between +/- infinity as
    defined in the Gurobi package
    """
    import gurobipy as grb
    n = len(v)
    for i in range(n):
        if v[i] >= 1e+20:
            v[i] = grb.GRB.INFINITY
        if v[i] <= -1e+20:
            v[i] = -grb.GRB.INFINITY