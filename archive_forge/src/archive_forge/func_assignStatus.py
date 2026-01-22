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
def assignStatus(self, status, sol_status=None):
    """
        Sets the status of the model after solving.
        :param status: code for the status of the model
        :param sol_status: code for the status of the solution
        :return:
        """
    if status not in const.LpStatus:
        raise const.PulpError('Invalid status code: ' + str(status))
    if sol_status is not None and sol_status not in const.LpSolution:
        raise const.PulpError('Invalid solution status code: ' + str(sol_status))
    self.status = status
    if sol_status is None:
        sol_status = const.LpStatusToSolution.get(status, const.LpSolutionNoSolutionFound)
    self.sol_status = sol_status
    return True