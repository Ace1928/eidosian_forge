import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
def displacement_log(self, displacement_vector, parameters):
    """Log the displacement"""
    if self.logfile is not None:
        lp = 'MINMODE:DISP: Parameters, different from the control:\n'
        mod_para = False
        for key in parameters:
            if parameters[key] != self.control.get_parameter(key):
                lp += 'MINMODE:DISP: %s = %s\n' % (str(key), str(parameters[key]))
                mod_para = True
        if mod_para:
            l = lp
        else:
            l = ''
        for k in range(len(displacement_vector)):
            l += 'MINMODE:DISP: %7i %15.8f %15.8f %15.8f\n' % (k, displacement_vector[k][0], displacement_vector[k][1], displacement_vector[k][2])
        self.logfile.write(l)
        self.logfile.flush()