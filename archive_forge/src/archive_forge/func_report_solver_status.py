from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def report_solver_status(self, status_code, status_message):
    self._asl.finalize_solution(status_code, status_message, self._primals, self._duals)