import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def test_expand_indexed_arc(self):

    def rule(m, i):
        return (m.c1[i], m.c2[i])
    m = ConcreteModel()
    m.x = Var(initialize=1, domain=Reals)
    m.y = Var(initialize=2, domain=Reals)
    m.c1 = Port([1, 2])
    m.c1[1].add(m.x, name='v')
    m.c1[2].add(m.y, name='t')
    m.z = Var(initialize=1, domain=Reals)
    m.w = Var(initialize=2, domain=Reals)
    m.c2 = Port([1, 2])
    m.c2[1].add(m.z, name='v')
    m.c2[2].add(m.w, name='t')
    m.eq = Arc([1, 2], rule=rule)
    TransformationFactory('network.expand_arcs').apply_to(m)
    self.assertFalse(m.eq.active)
    self.assertFalse(m.eq[1].active)
    self.assertFalse(m.eq[2].active)
    self.assertIs(m.eq.expanded_block, m.eq_expanded)
    self.assertIs(m.eq.expanded_block[1], m.eq_expanded[1])
    self.assertIs(m.eq.expanded_block[2], m.eq_expanded[2])
    os = StringIO()
    m.component('eq_expanded').pprint(ostream=os)
    self.assertEqual(os.getvalue(), 'eq_expanded : Size=2, Index={1, 2}, Active=True\n    eq_expanded[1] : Active=True\n        1 Constraint Declarations\n            v_equality : Size=1, Index=None, Active=True\n                Key  : Lower : Body  : Upper : Active\n                None :   0.0 : x - z :   0.0 :   True\n\n        1 Declarations: v_equality\n    eq_expanded[2] : Active=True\n        1 Constraint Declarations\n            t_equality : Size=1, Index=None, Active=True\n                Key  : Lower : Body  : Upper : Active\n                None :   0.0 : y - w :   0.0 :   True\n\n        1 Declarations: t_equality\n')