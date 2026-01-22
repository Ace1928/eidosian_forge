import collections
import itertools
import math
import numpy as np
import numpy.linalg as LA
import pytest
import cvxpy as cp
import cvxpy.interface as intf
from cvxpy.error import SolverError
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.settings import CVXOPT, ECOS, MOSEK, OSQP, ROBUST_KKTSOLVER, SCS
def run_atom(atom, problem, obj_val, solver, verbose: bool=False) -> None:
    assert problem.is_dcp()
    print(problem)
    if verbose:
        print(problem.objective)
        print(problem.constraints)
        print('solver', solver)
    if check_solver(problem, solver):
        tolerance = SOLVER_TO_TOL[solver]
        try:
            if solver == ROBUST_CVXOPT:
                result = problem.solve(solver=CVXOPT, verbose=verbose, kktsolver=ROBUST_KKTSOLVER)
            else:
                result = problem.solve(solver=solver, verbose=verbose)
        except SolverError as e:
            if (atom, solver) in KNOWN_SOLVER_ERRORS:
                return
            raise e
        if verbose:
            print(result)
            print(obj_val)
        assert -tolerance <= (result - obj_val) / (1 + np.abs(obj_val)) <= tolerance