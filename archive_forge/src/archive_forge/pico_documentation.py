import os.path
import subprocess
import pyomo.common
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.errors import ApplicationError
from pyomo.opt.base import ProblemFormat, ConverterError

        Run the external pico_convert utility
        