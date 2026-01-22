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
def test_subproblem_preprocessing_encounters_trivial_constraints(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    m.z = Var(bounds=(-10, 10))
    m.disjunction = Disjunction(expr=[[m.x == 0, m.z >= 4], [m.x + m.z <= 0]])
    m.cons = Constraint(expr=m.x * m.z <= 0)
    m.obj = Objective(expr=-m.z)
    m.disjunction.disjuncts[0].indicator_var.fix(True)
    m.disjunction.disjuncts[1].indicator_var.fix(False)
    SolverFactory('gdpopt.ric').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='fix_disjuncts')
    self.assertEqual(value(m.x), 0)
    self.assertEqual(value(m.z), 10)
    self.assertTrue(value(m.disjunction.disjuncts[0].indicator_var))
    self.assertFalse(value(m.disjunction.disjuncts[1].indicator_var))