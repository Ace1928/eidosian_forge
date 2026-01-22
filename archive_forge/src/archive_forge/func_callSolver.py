from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def callSolver(self, isMIP):
    """Solves the problem with cplex"""
    self.solveTime = -clock()
    self.solverModel.solve()
    self.solveTime += clock()