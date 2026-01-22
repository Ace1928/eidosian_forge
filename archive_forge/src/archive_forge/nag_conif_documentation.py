import warnings
import numpy as np
import scipy as sp
import cvxpy.settings as s
from cvxpy.constraints import SOC, NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        