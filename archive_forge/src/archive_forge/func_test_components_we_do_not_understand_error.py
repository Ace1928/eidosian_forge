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
def test_components_we_do_not_understand_error(self):
    m = self.makeModel()
    m.disj = Disjunction(expr=[m.x == 0, m.y >= 2])
    self.assertRaisesRegex(RuntimeError, 'Found active component %s of type %s. The Fourier-Motzkin Elimination transformation can only handle purely algebraic models. That is, only Sets, Params, Vars, Constraints, Expressions, Blocks, and Objectives may be active on the model.' % (m.disj.name, m.disj.type()), TransformationFactory('contrib.fourier_motzkin_elimination').apply_to, m, vars_to_eliminate=m.x)