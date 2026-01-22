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
def test_new_block_nameCollision(self):
    ct.check_transformation_block_name_collision(self, 'binary_multiplication')