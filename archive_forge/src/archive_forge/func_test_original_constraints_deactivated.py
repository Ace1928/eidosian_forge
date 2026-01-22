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
def test_original_constraints_deactivated(self):
    m = self.makeModel()
    TransformationFactory('contrib.fourier_motzkin_elimination').apply_to(m, vars_to_eliminate=m.lamb)
    self.assertFalse(m.primal1.active)
    self.assertFalse(m.dual1.active)
    self.assertFalse(m.dual2.active)
    self.assertFalse(m.bound_lambdas[1].active)
    self.assertFalse(m.bound_lambdas[2].active)
    self.assertFalse(m.bound_y.active)