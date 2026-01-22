from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def set_duals(self, duals):
    self._invalidate_duals_cache()
    np.copyto(self._duals_full, duals)
    np.compress(self._con_full_eq_mask, self._duals_full, out=self._duals_eq)
    np.compress(self._con_full_ineq_mask, self._duals_full, out=self._duals_ineq)