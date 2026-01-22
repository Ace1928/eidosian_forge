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
def test_check_convexity_conditions(self):
    c = self._object_factory()
    self.assertEqual(c.check_convexity_conditions(), True)
    c = self._object_factory()
    c.r1.domain_type = IntegerSet
    self.assertEqual(c.check_convexity_conditions(), False)
    self.assertEqual(c.check_convexity_conditions(relax=True), True)
    c = self._object_factory()
    c.r1.lb = None
    self.assertEqual(c.check_convexity_conditions(), False)
    c = self._object_factory()
    c.r1.lb = -1
    self.assertEqual(c.check_convexity_conditions(), False)
    c = self._object_factory()
    c.r2.domain_type = IntegerSet
    self.assertEqual(c.check_convexity_conditions(), False)
    self.assertEqual(c.check_convexity_conditions(relax=True), True)
    c = self._object_factory()
    c.r2.lb = None
    self.assertEqual(c.check_convexity_conditions(), False)
    c = self._object_factory()
    c.r2.lb = -1
    self.assertEqual(c.check_convexity_conditions(), False)
    c = self._object_factory()
    c.x[0].domain_type = IntegerSet
    self.assertEqual(c.check_convexity_conditions(), False)
    self.assertEqual(c.check_convexity_conditions(relax=True), True)
    c = self._object_factory()
    c.alpha.value = 0
    self.assertEqual(c.check_convexity_conditions(), False)
    c = self._object_factory()
    c.alpha.value = 1
    self.assertEqual(c.check_convexity_conditions(), False)