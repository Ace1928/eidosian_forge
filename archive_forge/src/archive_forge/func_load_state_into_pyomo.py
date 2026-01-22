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
def load_state_into_pyomo(self, bound_multipliers=None):
    primals = self.get_primals()
    variables = self.get_pyomo_variables()
    for var, val in zip(variables, primals):
        var.set_value(val)
    m = self.pyomo_model()
    model_suffixes = dict(pyo.suffix.active_import_suffix_generator(m))
    if 'dual' in model_suffixes:
        model_suffixes['dual'].clear()
    if 'ipopt_zL_out' in model_suffixes:
        model_suffixes['ipopt_zL_out'].clear()
        if bound_multipliers is not None:
            model_suffixes['ipopt_zL_out'].update(zip(variables, bound_multipliers[0]))
    if 'ipopt_zU_out' in model_suffixes:
        model_suffixes['ipopt_zU_out'].clear()
        if bound_multipliers is not None:
            model_suffixes['ipopt_zU_out'].update(zip(variables, bound_multipliers[1]))