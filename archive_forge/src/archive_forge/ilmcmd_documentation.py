import re
import sys
import os
import subprocess
import pyomo.common
from pyomo.common.errors import ApplicationError
import pyomo.opt.solver.shellcmd
from pyomo.opt.solver.shellcmd import SystemCallSolver
True if the solver is available