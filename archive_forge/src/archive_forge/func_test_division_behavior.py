import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import (
import pyomo.kernel
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.expression import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.objective import objective
from pyomo.core.kernel.block import block
def test_division_behavior(self):
    e = self._ctype_factory()
    e.expr = 2
    self.assertIs(type(e.expr), int)
    self.assertEqual((1 / e)(), 0.5)
    self.assertEqual((parameter(1) / e)(), 0.5)
    self.assertEqual(1 / e.expr, 0.5)