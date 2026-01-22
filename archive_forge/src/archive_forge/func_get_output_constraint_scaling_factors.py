import pyomo.environ as pyo
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
def get_output_constraint_scaling_factors(self):
    return np.asarray([10])