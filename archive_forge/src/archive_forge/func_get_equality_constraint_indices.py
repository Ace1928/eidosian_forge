import os
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import WriterFactory
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.common.env import CtypesEnviron
from ..sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from .external_grey_box import ExternalGreyBoxBlock
def get_equality_constraint_indices(self, constraints):
    """
        Return the list of equality indices for the constraints
        corresponding to the list of Pyomo constraints provided.

        Parameters
        ----------
        constraints : list of Pyomo Constraints or ConstraintData objects
        """
    indices = []
    for c in constraints:
        if c.is_indexed():
            for cd in c.values():
                con_eq_idx = self._condata_to_eq_idx[cd]
                indices.append(con_eq_idx)
        else:
            con_eq_idx = self._condata_to_eq_idx[c]
            indices.append(con_eq_idx)
    return indices