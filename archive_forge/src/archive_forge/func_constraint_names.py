from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def constraint_names(self):
    """Returns an ordered list with the names of all the constraints
        (corresponding to evaluate_constraints)"""
    return list(self._con_full_idx_to_name)