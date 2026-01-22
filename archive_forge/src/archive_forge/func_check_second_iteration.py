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
def check_second_iteration(self, model):
    transBlock = model.component('_pyomo_gdp_hull_reformulation_4')
    self.assertIsInstance(transBlock, Block)
    self.assertIsInstance(transBlock.component('relaxedDisjuncts'), Block)
    self.assertEqual(len(transBlock.relaxedDisjuncts), 2)
    hull = TransformationFactory('gdp.hull')
    if model.component('firstTerm') is None:
        firstTerm_cons = hull.get_transformed_constraints(model.component('firstTerm[1]').cons)
        secondTerm_cons = hull.get_transformed_constraints(model.component('secondTerm[1]').cons)
    else:
        firstTerm_cons = hull.get_transformed_constraints(model.firstTerm[1].cons)
        secondTerm_cons = hull.get_transformed_constraints(model.secondTerm[1].cons)
    self.assertEqual(len(firstTerm_cons), 1)
    self.assertIs(firstTerm_cons[0].parent_block(), transBlock.relaxedDisjuncts[0])
    self.assertEqual(len(secondTerm_cons), 1)
    self.assertIs(secondTerm_cons[0].parent_block(), transBlock.relaxedDisjuncts[1])
    orig = model.component('_pyomo_gdp_hull_reformulation')
    self.assertIsInstance(model.disjunctionList[1].algebraic_constraint, constraint._GeneralConstraintData)
    self.assertIsInstance(model.disjunctionList[0].algebraic_constraint, constraint._GeneralConstraintData)
    self.assertFalse(model.disjunctionList[1].active)
    self.assertFalse(model.disjunctionList[0].active)