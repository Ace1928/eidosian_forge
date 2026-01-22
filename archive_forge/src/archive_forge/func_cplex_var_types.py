from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def cplex_var_types(var):
    if var.cat == constants.LpInteger:
        return 'I'
    else:
        return 'C'