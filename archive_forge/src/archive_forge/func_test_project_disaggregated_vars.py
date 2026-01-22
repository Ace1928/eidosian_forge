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
def test_project_disaggregated_vars(self):
    """This is a little bit more of an integration test with GDP,
        but also an example of why FME is 'useful.' We will give a GDP,
        take hull relaxation, and then project out the disaggregated
        variables."""
    m, disaggregatedVars = self.create_hull_model()
    filtered = TransformationFactory('contrib.fourier_motzkin_elimination').create_using(m, vars_to_eliminate=disaggregatedVars)
    TransformationFactory('contrib.fourier_motzkin_elimination').apply_to(m, vars_to_eliminate=disaggregatedVars, constraint_filtering_callback=None, do_integer_arithmetic=True)
    constraints = m._pyomo_contrib_fme_transformation.projected_constraints
    self.check_hull_projected_constraints(m, constraints, [16, 12, 69, 71, 47, 60, 28, 1, 2, 3, 4])
    constraints = filtered._pyomo_contrib_fme_transformation.projected_constraints
    self.check_hull_projected_constraints(filtered, constraints, [8, 6, 20, 21, 13, 17, 9, 1, 2, 3, 4])