import pickle
from pyomo.common.dependencies import dill
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.base import constraint, ComponentUID
from pyomo.core.base.block import _BlockData
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
from io import StringIO
import random
import pyomo.opt
def check_solution_obeys_logical_constraints(self, transformation, m):
    trans = TransformationFactory('gdp.%s' % transformation)
    m.p.deactivate()
    m.bwahaha.deactivate()
    no_logic = trans.create_using(m)
    results = SolverFactory(linear_solvers[0]).solve(no_logic)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(value(no_logic.x), 2.5)
    m.p.activate()
    m.bwahaha.activate()
    trans.apply_to(m)
    results = SolverFactory(linear_solvers[0]).solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(value(m.x), 8)