import pickle
import math
import pyomo.common.unittest as unittest
from pyomo.kernel import pprint, IntegerSet
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.variable import variable, variable_tuple
from pyomo.core.kernel.block import block
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
from pyomo.core.kernel.conic import (
def test_as_domain(self):
    ret = dual_power.as_domain(r1=3, r2=4, x=[1, 2], alpha=0.5)
    self.assertIs(type(ret), block)
    q, c, r1, r2, x = (ret.q, ret.c, ret.r1, ret.r2, ret.x)
    self.assertEqual(q.check_convexity_conditions(), True)
    self.assertIs(type(q), dual_power)
    self.assertIs(type(r1), variable)
    self.assertIs(type(r2), variable)
    self.assertIs(type(x), variable_tuple)
    self.assertEqual(len(x), 2)
    self.assertIs(type(c), constraint_tuple)
    self.assertEqual(len(c), 4)
    self.assertEqual(c[0].rhs, 3)
    r1.value = 3
    self.assertEqual(c[0].slack, 0)
    r1.value = None
    self.assertEqual(c[1].rhs, 4)
    r2.value = 4
    self.assertEqual(c[1].slack, 0)
    r2.value = None
    self.assertEqual(c[2].rhs, 1)
    x[0].value = 1
    self.assertEqual(c[2].slack, 0)
    x[0].value = None
    self.assertEqual(c[3].rhs, 2)
    x[1].value = 2
    self.assertEqual(c[3].slack, 0)
    x[1].value = None