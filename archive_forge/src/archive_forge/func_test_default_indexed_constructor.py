import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_default_indexed_constructor(self):
    m = ConcreteModel()
    m.c1 = Arc([1, 2, 3])
    self.assertEqual(len(m.c1), 0)
    self.assertIs(m.c1.ctype, Arc)
    m = AbstractModel()
    m.c1 = Arc([1, 2, 3])
    self.assertEqual(len(m.c1), 0)
    self.assertIs(m.c1.ctype, Arc)
    inst = m.create_instance()
    self.assertEqual(len(m.c1), 0)
    self.assertIs(m.c1.ctype, Arc)