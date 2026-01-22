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
def test_init_non_NumericValue(self):
    types = [None, 1, 1.1, True, '']
    if numpy_available:
        types.extend([numpy.float32(1), numpy.bool_(True), numpy.int32(1)])
    types.append(block())
    types.append(block)
    for obj in types:
        self.assertEqual(noclone(obj), obj)
        self.assertIs(type(noclone(obj)), type(obj))