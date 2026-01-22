from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def check_block_c1_transformed(self, m, transBlock):
    self.assertFalse(m.block.c1.active)
    self.assertIs(m.a.get_associated_binary(), transBlock.auxiliary_vars[1])
    self.assertIs(m.b[1].get_associated_binary(), transBlock.auxiliary_vars[2])
    self.check_and_constraints(transBlock.auxiliary_vars[1], transBlock.auxiliary_vars[2], transBlock.auxiliary_vars[3], transBlock)