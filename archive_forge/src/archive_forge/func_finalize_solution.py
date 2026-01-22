from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def finalize_solution(self, ampl_solve_status_num, msg, x, lam):
    b_msg = msg.encode('utf-8')
    self.ASLib.EXTERNAL_AmplInterface_finalize_solution(self._obj, ampl_solve_status_num, b_msg, x, len(x), lam, len(lam))