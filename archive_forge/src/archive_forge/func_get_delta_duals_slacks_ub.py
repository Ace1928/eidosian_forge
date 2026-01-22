from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
def get_delta_duals_slacks_ub(self):
    res = (self._barrier + self._duals_slacks_ub * self._delta_slacks) / (self._nlp.ineq_ub() - self._slacks) - self._duals_slacks_ub
    return res