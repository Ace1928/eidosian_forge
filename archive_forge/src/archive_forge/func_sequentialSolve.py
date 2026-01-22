from collections import Counter
import sys
import warnings
from time import time
from .apis import LpSolverDefault, PULP_CBC_CMD
from .apis.core import clock
from .utilities import value
from . import constants as const
from . import mps_lp as mpslp
import logging
import re
def sequentialSolve(self, objectives, absoluteTols=None, relativeTols=None, solver=None, debug=False):
    """
        Solve the given Lp problem with several objective functions.

        This function sequentially changes the objective of the problem
        and then adds the objective function as a constraint

        :param objectives: the list of objectives to be used to solve the problem
        :param absoluteTols: the list of absolute tolerances to be applied to
           the constraints should be +ve for a minimise objective
        :param relativeTols: the list of relative tolerances applied to the constraints
        :param solver: the specific solver to be used, defaults to the default solver.

        """
    if not solver:
        solver = self.solver
    if not solver:
        solver = LpSolverDefault
    if not absoluteTols:
        absoluteTols = [0] * len(objectives)
    if not relativeTols:
        relativeTols = [1] * len(objectives)
    self.startClock()
    statuses = []
    for i, (obj, absol, rel) in enumerate(zip(objectives, absoluteTols, relativeTols)):
        self.setObjective(obj)
        status = solver.actualSolve(self)
        statuses.append(status)
        if debug:
            self.writeLP(f'{i}Sequence.lp')
        if self.sense == const.LpMinimize:
            self += (obj <= value(obj) * rel + absol, f'Sequence_Objective_{i}')
        elif self.sense == const.LpMaximize:
            self += (obj >= value(obj) * rel + absol, f'Sequence_Objective_{i}')
    self.stopClock()
    self.solver = solver
    return statuses