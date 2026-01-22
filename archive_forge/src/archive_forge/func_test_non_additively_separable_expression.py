import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
def test_non_additively_separable_expression(self):
    m = models.makeNonQuadraticNonlinearGDP()
    m.disjunction.disjuncts[0].another_constraint = Constraint(expr=m.x[1] ** 3 <= 0.5)
    TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)
    b = m.component('_pyomo_gdp_partition_disjuncts_reformulation')
    disj1 = b.disjunction.disjuncts[0]
    self.assertEqual(len(disj1.component_map(Constraint)), 2)
    self.assertEqual(len(disj1.component_map(Var)), 4)
    self.assertEqual(len(disj1.component_map(Constraint)), 2)
    aux_vars1 = disj1.component('disjunction_disjuncts[0].constraint[1]_aux_vars')
    aux_vars2 = disj1.component('disjunction_disjuncts[0].another_constraint_aux_vars')
    self.assertEqual(len(aux_vars2), 1)
    self.assertEqual(aux_vars2[0].lb, -8)
    self.assertEqual(aux_vars2[0].ub, 216)
    cons = disj1.component('disjunction_disjuncts[0].another_constraint')
    self.assertEqual(len(cons), 1)
    cons = cons[0]
    self.assertIsNone(cons.lower)
    self.assertEqual(cons.upper, 0.5)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertEqual(repn.linear_coefs[0], 1)
    self.assertIs(repn.linear_vars[0], aux_vars2[0])
    cons = b.component('disjunction_disjuncts[0].another_constraint_split_constraints')
    self.assertEqual(len(cons), 1)
    cons = cons[0]
    self.assertIsNone(cons.lower)
    self.assertEqual(cons.upper, 0)
    repn = generate_standard_repn(cons.body)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertEqual(repn.linear_coefs[0], -1)
    self.assertIs(repn.linear_vars[0], aux_vars2[0])
    self.assertEqual(len(repn.nonlinear_vars), 1)
    self.assertIs(repn.nonlinear_vars[0], m.x[1])
    nonlinear = repn.nonlinear_expr
    self.assertIsInstance(nonlinear, EXPR.PowExpression)
    self.assertIs(nonlinear.args[0], m.x[1])
    self.assertEqual(nonlinear.args[1], 3)