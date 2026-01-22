import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
def test_log_disclaimer(self):
    """
        Test logging of PyROS solver disclaimer messages.
        """
    pyros_solver = SolverFactory('pyros')
    with LoggingIntercept(level=logging.INFO) as LOG:
        pyros_solver._log_disclaimer(logger=logger, level=logging.INFO)
    disclaimer_msgs = LOG.getvalue()
    disclaimer_msg_lines = disclaimer_msgs.split('\n')[:-1]
    self.assertEqual(len(disclaimer_msg_lines), 5, msg='PyROS solver disclaimer message does not containthe expected number of lines.')
    self.assertRegex(disclaimer_msg_lines[0], '=.* DISCLAIMER .*=')
    self.assertEqual(disclaimer_msg_lines[-1], '=' * 78)
    self.assertRegex(' '.join(disclaimer_msg_lines[1:-1]), 'PyROS is still under development.*ticket at.*')