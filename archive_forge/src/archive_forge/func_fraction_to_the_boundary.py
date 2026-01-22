from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum
def fraction_to_the_boundary(self):
    return fraction_to_the_boundary(self.interface, 1 - self._barrier_parameter)