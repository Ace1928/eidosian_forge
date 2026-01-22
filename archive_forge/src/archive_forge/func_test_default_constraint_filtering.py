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
def test_default_constraint_filtering(self):
    m = self.makeModel()
    TransformationFactory('contrib.fourier_motzkin_elimination').apply_to(m, vars_to_eliminate=m.lamb)
    self.check_projected_constraints(m, self.filtered_indices)
    constraints = m._pyomo_contrib_fme_transformation.projected_constraints
    self.assertEqual(len(constraints), 4)