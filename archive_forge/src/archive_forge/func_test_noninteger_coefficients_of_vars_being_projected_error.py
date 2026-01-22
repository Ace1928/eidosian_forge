import os
from os.path import abspath, dirname
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet
from pyomo.core import (
from pyomo.core.base import TransformationFactory
from pyomo.core.expr import log
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.gdp import Disjunction, Disjunct
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.opt import SolverFactory, check_available_solvers
import pyomo.contrib.fme.fourier_motzkin_elimination
from io import StringIO
import logging
import random
@unittest.skipIf(not 'glpk' in solvers, 'glpk not available')
def test_noninteger_coefficients_of_vars_being_projected_error(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 9))
    m.y = Var(bounds=(-5, 5))
    m.c1 = Constraint(expr=2 * m.x + 0.5 * m.y >= 2)
    m.c2 = Constraint(expr=0.25 * m.y >= 0.5 * m.x)
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    self.assertRaisesRegex(ValueError, 'The do_integer_arithmetic flag was set to True, but the coefficient of x is non-integer within the specified tolerance, with value -0.5. \nPlease set do_integer_arithmetic=False, increase integer_tolerance, or make your data integer.', fme.apply_to, m, vars_to_eliminate=m.x, do_integer_arithmetic=True)