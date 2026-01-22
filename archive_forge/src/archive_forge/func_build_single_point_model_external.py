import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
import scipy.sparse as spa
import numpy as np
import math
def build_single_point_model_external(m):
    ex_model = UAModelExternal()
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_model)