from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum
def update_barrier_parameter(self):
    self._barrier_parameter = max(self._minimum_barrier_parameter, min(0.5 * self._barrier_parameter, self._barrier_parameter ** 1.5))