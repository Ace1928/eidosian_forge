from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def check_trans_block_disjunctions_of_disjunct_datas(self, m):
    transBlock1 = m.component('_pyomo_gdp_hull_reformulation')
    self.assertIsInstance(transBlock1, Block)
    self.assertIsInstance(transBlock1.component('relaxedDisjuncts'), Block)
    self.assertEqual(len(transBlock1.relaxedDisjuncts), 4)
    hull = TransformationFactory('gdp.hull')
    firstTerm2 = transBlock1.relaxedDisjuncts[2]
    self.assertIs(firstTerm2, m.firstTerm[2].transformation_block)
    self.assertIsInstance(firstTerm2.disaggregatedVars.component('x'), Var)
    constraints = hull.get_transformed_constraints(m.firstTerm[2].cons)
    self.assertEqual(len(constraints), 1)
    cons = constraints[0]
    self.assertIs(cons.parent_block(), firstTerm2)
    dis_x = hull.get_disaggregated_var(m.x, m.firstTerm[2])
    cons = hull.get_var_bounds_constraint(dis_x)
    self.assertIsInstance(cons, Constraint)
    self.assertIs(cons.parent_block(), firstTerm2)
    self.assertEqual(len(cons), 2)
    secondTerm2 = transBlock1.relaxedDisjuncts[3]
    self.assertIs(secondTerm2, m.secondTerm[2].transformation_block)
    self.assertIsInstance(secondTerm2.disaggregatedVars.component('x'), Var)
    constraints = hull.get_transformed_constraints(m.secondTerm[2].cons)
    self.assertEqual(len(constraints), 1)
    cons = constraints[0]
    self.assertIs(cons.parent_block(), secondTerm2)
    dis_x = hull.get_disaggregated_var(m.x, m.secondTerm[2])
    cons = hull.get_var_bounds_constraint(dis_x)
    self.assertIsInstance(cons, Constraint)
    self.assertIs(cons.parent_block(), secondTerm2)
    self.assertEqual(len(cons), 2)
    firstTerm1 = transBlock1.relaxedDisjuncts[0]
    self.assertIs(firstTerm1, m.firstTerm[1].transformation_block)
    self.assertIsInstance(firstTerm1.disaggregatedVars.component('x'), Var)
    self.assertTrue(firstTerm1.disaggregatedVars.x.is_fixed())
    self.assertEqual(value(firstTerm1.disaggregatedVars.x), 0)
    constraints = hull.get_transformed_constraints(m.firstTerm[1].cons)
    self.assertEqual(len(constraints), 1)
    cons = constraints[0]
    self.assertIs(cons.parent_block(), firstTerm1.disaggregatedVars)
    dis_x = hull.get_disaggregated_var(m.x, m.firstTerm[1])
    cons = hull.get_var_bounds_constraint(dis_x)
    self.assertIsInstance(cons, Constraint)
    self.assertIs(cons.parent_block(), firstTerm1)
    self.assertEqual(len(cons), 2)
    secondTerm1 = transBlock1.relaxedDisjuncts[1]
    self.assertIs(secondTerm1, m.secondTerm[1].transformation_block)
    self.assertIsInstance(secondTerm1.disaggregatedVars.component('x'), Var)
    constraints = hull.get_transformed_constraints(m.secondTerm[1].cons)
    self.assertEqual(len(constraints), 1)
    cons = constraints[0]
    self.assertIs(cons.parent_block(), secondTerm1)
    dis_x = hull.get_disaggregated_var(m.x, m.secondTerm[1])
    cons = hull.get_var_bounds_constraint(dis_x)
    self.assertIsInstance(cons, Constraint)
    self.assertIs(cons.parent_block(), secondTerm1)
    self.assertEqual(len(cons), 2)