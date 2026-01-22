import os
import shutil
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.core.base.external import (
from pyomo.core.base.units_container import pint_available, units
from pyomo.core.expr.numeric_expr import (
from pyomo.opt import check_available_solvers
def test_unknown_library(self):
    m = ConcreteModel()
    with LoggingIntercept() as LOG:
        m.ef = ExternalFunction(library='unknown_pyomo_external_testing_function', function='f')
    self.assertEqual(LOG.getvalue(), 'Defining AMPL external function, but cannot locate specified library "unknown_pyomo_external_testing_function"\n')