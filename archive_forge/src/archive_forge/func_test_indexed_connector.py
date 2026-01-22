import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
def test_indexed_connector(self):
    m = ConcreteModel()
    m.x = Var(initialize=1, domain=Reals)
    m.y = Var(initialize=2, domain=Reals)
    m.c = Connector([1, 2])
    m.c[1].add(m.x, name='v')
    m.c[2].add(m.y, name='v')
    m.eq = Constraint(expr=m.c[1] == m.c[2])
    TransformationFactory('core.expand_connectors').apply_to(m)
    os = StringIO()
    m.component('eq.expanded').pprint(ostream=os)
    self.assertEqual(os.getvalue(), 'eq.expanded : Size=1, Index={1}, Active=True\n    Key : Lower : Body  : Upper : Active\n      1 :   0.0 : x - y :   0.0 :   True\n')