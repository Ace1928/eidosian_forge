from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_transform_model(self):
    m = self.make_model()
    TransformationFactory('contrib.logical_to_disjunctive').apply_to(m)
    self.assertFalse(m.c1.active)
    transBlock = m._logical_to_disjunctive
    self.assertEqual(len(transBlock.auxiliary_vars), 3)
    self.assertEqual(len(transBlock.transformed_constraints), 1)
    self.assertEqual(len(transBlock.auxiliary_disjuncts), 2)
    self.assertEqual(len(transBlock.auxiliary_disjunctions), 1)
    a = m._logical_to_disjunctive.auxiliary_vars[1]
    b1 = m._logical_to_disjunctive.auxiliary_vars[2]
    b2 = m._logical_to_disjunctive.auxiliary_vars[3]
    assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[0].constraint.expr, a + b1 + b2 <= 1 + m.p2[1])
    assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[1].constraint.expr, a + b1 + b2 >= 1 + m.p2[1] + 1)
    assertExpressionsEqual(self, transBlock.transformed_constraints[1].expr, transBlock.auxiliary_disjuncts[0].binary_indicator_var >= 1)
    transBlock = m.block._logical_to_disjunctive
    self.assertEqual(len(transBlock.auxiliary_vars), 2)
    self.assertEqual(len(transBlock.transformed_constraints), 8)
    self.assertEqual(len(transBlock.auxiliary_disjuncts), 2)
    self.assertEqual(len(transBlock.auxiliary_disjunctions), 1)
    self.check_and_constraints(a, b1, transBlock.auxiliary_vars[1], transBlock)
    self.check_block_exactly(a, b1, b2, transBlock.auxiliary_vars[2], transBlock)