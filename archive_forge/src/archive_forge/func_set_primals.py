from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def set_primals(self, primals):
    self._invalidate_primals_cache()
    np.copyto(self._primals, primals)