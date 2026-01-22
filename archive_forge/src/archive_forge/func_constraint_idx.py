from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def constraint_idx(self, con_name):
    """
        Returns the index of the constraint named con_name
        (corresponding to the order returned by evaluate_constraints)

        Parameters
        ----------
        con_name: str
            Name of constraint

        Returns
        -------
        int
        """
    return self._name_to_con_full_idx[con_name]