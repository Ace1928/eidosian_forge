from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
def set_primal_dual_kkt_solution(self, sol):
    self._delta_primals = sol.get_block(0)
    self._delta_slacks = sol.get_block(1)
    self._delta_duals_eq = sol.get_block(2)
    self._delta_duals_ineq = sol.get_block(3)