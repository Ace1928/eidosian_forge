import os
from pyomo.environ import SolverFactory
from pyomo.common.tempfiles import TempfileManager
def set_dot_sens_options(self, **options):
    for key, val in options.items():
        self._dot_sens.options[key] = val