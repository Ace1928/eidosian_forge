from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def get_ineq_constraints_scaling(self):
    constraints_scaling = self.get_constraints_scaling()
    if constraints_scaling is not None:
        return np.compress(self._con_full_ineq_mask, constraints_scaling)
    return None