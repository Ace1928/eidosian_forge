import warnings
import numpy as np
import scipy  # For version checks
import cvxpy.settings as s
from cvxpy.constraints import NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version
Returns the solution to the original problem given the inverse_data.
        