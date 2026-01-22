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
def get_pyomo_equality_constraints(self):
    """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the equality constraints.
        """
    idx_to_condata = {i: c for c, i in self._condata_to_eq_idx.items()}
    return [idx_to_condata[i] for i in range(len(idx_to_condata))]