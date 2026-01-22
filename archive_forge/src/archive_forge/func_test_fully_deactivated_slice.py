import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_fully_deactivated_slice(self):
    m = ConcreteModel()
    m.time = Set(initialize=[0, 1, 2, 3])
    m.b = Block(m.time)
    for t in m.time:
        m.b[t].v = Var()
    m.b[:].deactivate()
    sets = (m.time,)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=True)
    self.assertEqual(len(sets_list), 0)
    self.assertEqual(len(comps_list), 0)