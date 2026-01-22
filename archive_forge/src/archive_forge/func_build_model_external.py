from pyomo.common.dependencies import numpy as np
import pyomo.common.dependencies.scipy.sparse as scipy_sparse
from pyomo.common.dependencies import attempt_import
def build_model_external(m):
    ex_model = GreyBoxModel(initial={'X1': 0, 'X2': 0, 'Y1': 0, 'Y2': 1, 'Y3': 1})
    m.egb = egb.ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_model)