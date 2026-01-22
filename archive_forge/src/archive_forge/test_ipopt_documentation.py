import pyomo.environ as pyo
from pyomo.common.fileutils import ExecutableData
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver.ipopt import IpoptConfig
from pyomo.contrib.solver.factory import SolverFactory
from pyomo.common import unittest

TODO:
    - Test unique configuration options
    - Test unique results options
    - Ensure that `*.opt` file is only created when needed
    - Ensure options are correctly parsing to env or opt file
    - Failures at appropriate times
