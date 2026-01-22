from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def eval_jac_g(self, x, jac_g_values):
    assert x.size == self._nx, 'Error: Dimension mismatch.'
    assert jac_g_values.size == self._nnz_jac_g, 'Error: Dimension mismatch.'
    xeval = x.astype(np.double, casting='safe', copy=False)
    jac_eval = jac_g_values.astype(np.double, casting='safe', copy=False)
    res = self.ASLib.EXTERNAL_AmplInterface_eval_jac_g(self._obj, xeval, self._nx, jac_eval, self._nnz_jac_g)
    if not res:
        raise PyNumeroEvaluationError('Error in AMPL evaluation')