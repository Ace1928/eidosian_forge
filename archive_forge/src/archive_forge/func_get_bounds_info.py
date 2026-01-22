from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def get_bounds_info(self, xl, xu, gl, gu):
    x_l = xl.astype(np.double, casting='safe', copy=False)
    x_u = xu.astype(np.double, casting='safe', copy=False)
    g_l = gl.astype(np.double, casting='safe', copy=False)
    g_u = gu.astype(np.double, casting='safe', copy=False)
    nx = len(x_l)
    ng = len(g_l)
    assert nx == len(x_u), 'lower and upper bound x vectors must be the same size'
    assert ng == len(g_u), 'lower and upper bound g vectors must be the same size'
    self.ASLib.EXTERNAL_AmplInterface_get_bounds_info(self._obj, x_l, x_u, nx, g_l, g_u, ng)