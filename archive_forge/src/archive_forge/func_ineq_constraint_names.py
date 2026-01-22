from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def ineq_constraint_names(self):
    """Returns ordered list with names of inequality constraints only
        (corresponding to evaluate_ineq_constraints)"""
    return list(self._con_ineq_idx_to_name)