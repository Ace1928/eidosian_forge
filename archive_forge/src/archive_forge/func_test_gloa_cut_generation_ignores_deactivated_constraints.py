from contextlib import redirect_stdout
from io import StringIO
import logging
from math import fabs
from os.path import join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from pyomo.contrib.gdpopt.create_oa_subproblems import (
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.gdpopt.util import is_feasible, time_code
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.tests import models
from pyomo.opt import TerminationCondition
def test_gloa_cut_generation_ignores_deactivated_constraints(self):
    m = self.get_GDP_on_block()
    m.b.disjunction.disjuncts[0].indicator_var.fix(True)
    m.b.disjunction.disjuncts[1].indicator_var.fix(False)
    m.b.disjunction.disjuncts[2].indicator_var.fix(False)
    m.b.deactivate()
    m.disjunction.disjuncts[0].indicator_var.fix(False)
    m.disjunction.disjuncts[1].indicator_var.fix(True)
    m.disjunction.disjuncts[2].indicator_var.fix(False)
    add_util_block(m)
    util_block = m._gdpopt_cuts
    add_disjunct_list(util_block)
    add_constraints_by_disjunct(util_block)
    add_global_constraint_list(util_block)
    TransformationFactory('gdp.bigm').apply_to(m)
    config = ConfigDict()
    config.declare('integer_tolerance', ConfigValue(1e-06))
    gloa = SolverFactory('gdpopt.gloa')
    constraints = list(gloa._get_active_untransformed_constraints(util_block, config))
    self.assertEqual(len(constraints), 2)
    c1 = constraints[0]
    c2 = constraints[1]
    self.assertIs(c1.body, m.x)
    self.assertEqual(c1.lower, 0)
    self.assertEqual(c1.upper, 0)
    self.assertIs(c2.body, m.y)
    self.assertEqual(c2.lower, 0)
    self.assertIsNone(c2.upper)