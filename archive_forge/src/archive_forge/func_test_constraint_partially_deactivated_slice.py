import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_constraint_partially_deactivated_slice(self):
    m = ConcreteModel()
    m.time = Set(initialize=[0, 1, 2, 3])
    m.b = Block(m.time)
    for t in m.time:
        m.b[t].v = Var()

    def c2_rule(m, t):
        return m.b[t].v == 2
    m.c2 = Constraint(m.time, rule=c2_rule)
    m.c2[0].deactivate()
    m.c2[1].deactivate()
    sets = (m.time,)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Constraint, active=True)
    self.assertEqual(len(sets_list), 1)
    self.assertEqual(len(sets_list[0]), 1)
    self.assertIs(sets_list[0][0], m.time)
    self.assertIs(sets_list[0][0], m.time)
    self.assertEqual(len(comps_list), 1)
    self.assertEqual(len(comps_list[0]), 1)
    self.assertEqual(ComponentUID(comps_list[0][0].referent), ComponentUID(m.c2[:]))