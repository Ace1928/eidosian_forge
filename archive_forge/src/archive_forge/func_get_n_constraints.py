from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def get_n_constraints(self):
    return self.ASLib.EXTERNAL_AmplInterface_n_constraints(self._obj)