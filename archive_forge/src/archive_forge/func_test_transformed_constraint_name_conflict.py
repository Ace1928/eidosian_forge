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
def test_transformed_constraint_name_conflict(self):
    m = self.makeModel()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    transBlock = m.disj1.transformation_block
    self.assertEqual(len(transBlock.component_map(Constraint)), 3)
    self.assertIs(hull.get_transformed_constraints(m.disj1.b.any_index['local'])[0].parent_block(), transBlock)
    self.assertIs(hull.get_transformed_constraints(m.disj1.b.any_index['nonlin-ub'])[0].parent_block(), transBlock)
    self.assertIs(hull.get_transformed_constraints(m.disj1.component('b.any_index'))[0].parent_block(), transBlock)