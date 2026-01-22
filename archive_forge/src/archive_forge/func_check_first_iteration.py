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
def check_first_iteration(self, model):
    transBlock = model.component('_pyomo_gdp_hull_reformulation')
    self.assertIsInstance(transBlock, Block)
    self.assertIsInstance(transBlock.component('disjunctionList_xor'), Constraint)
    self.assertEqual(len(transBlock.disjunctionList_xor), 1)
    self.assertFalse(model.disjunctionList[0].active)
    hull = TransformationFactory('gdp.hull')
    if model.component('firstTerm') is None:
        firstTerm_cons = hull.get_transformed_constraints(model.component('firstTerm[0]').cons)
        secondTerm_cons = hull.get_transformed_constraints(model.component('secondTerm[0]').cons)
    else:
        firstTerm_cons = hull.get_transformed_constraints(model.firstTerm[0].cons)
        secondTerm_cons = hull.get_transformed_constraints(model.secondTerm[0].cons)
    self.assertIsInstance(transBlock.relaxedDisjuncts, Block)
    self.assertEqual(len(transBlock.relaxedDisjuncts), 2)
    self.assertIsInstance(transBlock.relaxedDisjuncts[0].disaggregatedVars.x, Var)
    self.assertTrue(transBlock.relaxedDisjuncts[0].disaggregatedVars.x.is_fixed())
    self.assertEqual(value(transBlock.relaxedDisjuncts[0].disaggregatedVars.x), 0)
    self.assertEqual(len(firstTerm_cons), 1)
    self.assertIs(firstTerm_cons[0].parent_block(), transBlock.relaxedDisjuncts[0].disaggregatedVars)
    self.assertIsInstance(transBlock.relaxedDisjuncts[0].x_bounds, Constraint)
    self.assertEqual(len(transBlock.relaxedDisjuncts[0].x_bounds), 2)
    self.assertIsInstance(transBlock.relaxedDisjuncts[1].disaggregatedVars.x, Var)
    self.assertFalse(transBlock.relaxedDisjuncts[1].disaggregatedVars.x.is_fixed())
    self.assertEqual(len(secondTerm_cons), 1)
    self.assertIs(secondTerm_cons[0].parent_block(), transBlock.relaxedDisjuncts[1])
    self.assertIsInstance(transBlock.relaxedDisjuncts[1].x_bounds, Constraint)
    self.assertEqual(len(transBlock.relaxedDisjuncts[1].x_bounds), 2)