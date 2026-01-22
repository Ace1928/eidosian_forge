import pickle
import pyomo.common.unittest as unittest
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.sos import ISOS, sos, sos1, sos2, sos_dict, sos_tuple, sos_list
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
def test_bad_weights(self):
    v = variable()
    with self.assertRaises(ValueError):
        s = sos([v], weights=[v])
    v.fix(1.0)
    with self.assertRaises(ValueError):
        s = sos([v], weights=[v])
    e = expression()
    with self.assertRaises(ValueError):
        s = sos([v], weights=[e])
    de = data_expression()
    s = sos([v], weights=[de])
    p = parameter()
    s = sos([v], weights=[p])