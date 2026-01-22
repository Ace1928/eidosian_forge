from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def check_block_transformed(self, m):
    self.assertFalse(m.block.c2.active)
    transBlock = m.block._logical_to_disjunctive
    self.assertEqual(len(transBlock.auxiliary_vars), 5)
    self.assertEqual(len(transBlock.transformed_constraints), 8)
    self.assertEqual(len(transBlock.auxiliary_disjuncts), 2)
    self.assertEqual(len(transBlock.auxiliary_disjunctions), 1)
    self.check_block_c1_transformed(m, transBlock)
    self.assertIs(m.b[2].get_associated_binary(), transBlock.auxiliary_vars[4])
    z4 = transBlock.auxiliary_vars[5]
    a = transBlock.auxiliary_vars[1]
    b1 = transBlock.auxiliary_vars[2]
    b2 = transBlock.auxiliary_vars[4]
    self.check_block_exactly(a, b1, b2, z4, transBlock)