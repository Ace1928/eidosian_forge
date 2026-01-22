import os
import numpy as np
import logging
from scipy.sparse import coo_matrix, identity
from pyomo.common.deprecation import deprecated
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.utils import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedNLP
def has_hessian_support(self):
    return self._has_hessian_support