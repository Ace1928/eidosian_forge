from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def check_block_exactly(self, a, b1, b2, z4, transBlock):
    m = transBlock.model()
    assertExpressionsEqual(self, transBlock.transformed_constraints[5].expr, 1 - z4 + b2 + b1 >= 1)
    assertExpressionsEqual(self, transBlock.transformed_constraints[6].expr, z4 + (1 - b2) >= 1)
    assertExpressionsEqual(self, transBlock.transformed_constraints[7].expr, z4 + (1 - b1) >= 1)
    assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[0].constraint.expr, a + b1 + z4 == m.p2[2])
    assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[1].disjunction.disjuncts[0].constraint[1].expr, a + b1 + z4 <= m.p2[2] - 1)
    assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[1].disjunction.disjuncts[1].constraint[1].expr, a + b1 + z4 >= m.p2[2] + 1)
    assertExpressionsEqual(self, transBlock.transformed_constraints[8].expr, transBlock.auxiliary_disjuncts[0].binary_indicator_var >= 1)