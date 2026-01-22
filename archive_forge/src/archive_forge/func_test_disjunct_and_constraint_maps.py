import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
def test_disjunct_and_constraint_maps(self):
    """Tests the actual data structures used to store the maps."""
    m = models.makeTwoTermDisj()
    binary_multiplication = TransformationFactory('gdp.binary_multiplication')
    binary_multiplication.apply_to(m)
    disjBlock = m._pyomo_gdp_binary_multiplication_reformulation.relaxedDisjuncts
    oldblock = m.component('d')
    for i in [0, 1]:
        self.assertIs(oldblock[i].transformation_block, disjBlock[i])
        self.assertIs(binary_multiplication.get_src_disjunct(disjBlock[i]), oldblock[i])
    c1_list = binary_multiplication.get_transformed_constraints(oldblock[1].c1)
    self.assertEqual(len(c1_list), 1)
    self.assertIs(c1_list[0].parent_block(), disjBlock[1])
    self.assertIs(binary_multiplication.get_src_constraint(c1_list[0]), oldblock[1].c1)
    c2_list = binary_multiplication.get_transformed_constraints(oldblock[1].c2)
    self.assertEqual(len(c2_list), 1)
    self.assertIs(c2_list[0].parent_block(), disjBlock[1])
    self.assertIs(binary_multiplication.get_src_constraint(c2_list[0]), oldblock[1].c2)
    c_list = binary_multiplication.get_transformed_constraints(oldblock[0].c)
    self.assertEqual(len(c_list), 1)
    self.assertIs(c_list[0].parent_block(), disjBlock[0])
    self.assertIs(binary_multiplication.get_src_constraint(c_list[0]), oldblock[0].c)