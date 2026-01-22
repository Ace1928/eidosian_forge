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
def test_do_not_transform_deactivated_constraintDatas(self):
    m = models.makeTwoTermDisj_IndexedConstraints()
    m.a[1].setlb(0)
    m.a[1].setub(100)
    m.a[2].setlb(0)
    m.a[2].setub(100)
    m.b.simpledisj1.c[1].deactivate()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    log = StringIO()
    with LoggingIntercept(log, 'pyomo.gdp', logging.ERROR):
        self.assertRaisesRegex(KeyError, '.*b.simpledisj1.c\\[1\\]', hull.get_transformed_constraints, m.b.simpledisj1.c[1])
    self.assertRegex(log.getvalue(), ".*Constraint 'b.simpledisj1.c\\[1\\]' has not been transformed.")
    transformed = hull.get_transformed_constraints(m.b.simpledisj1.c[2])
    self.assertEqual(len(transformed), 1)
    disaggregated_a2 = hull.get_disaggregated_var(m.a[2], m.b.simpledisj1)
    self.assertIs(transformed[0], disaggregated_a2)
    self.assertIsInstance(disaggregated_a2, Var)
    self.assertTrue(disaggregated_a2.is_fixed())
    self.assertEqual(value(disaggregated_a2), 0)
    transformed = hull.get_transformed_constraints(m.b.simpledisj2.c[1])
    self.assertEqual(len(transformed), 1)
    self.assertIs(transformed[0].parent_block(), m.b.simpledisj2.transformation_block)
    transformed = hull.get_transformed_constraints(m.b.simpledisj2.c[2])
    self.assertEqual(len(transformed), 1)
    self.assertIs(transformed[0].parent_block(), m.b.simpledisj2.transformation_block)