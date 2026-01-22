from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum
def set_linear_solver(self, linear_solver):
    """This method exists to hopefully make it easy to try the same IP
        algorithm with different linear solvers.
        Subclasses may have linear-solver specific methods, in which case
        this should not be called.

        Hopefully the linear solver interface can be standardized such that
        this is not a problem. (Need a generalized method for set_options)
        """
    self.linear_solver = linear_solver