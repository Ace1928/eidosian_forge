import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_stop_iteration(self):
    """
        StopIteration is raised if we create an empty slice somewhere
        along the line. It is an open question what we should do in the
        case of an empty slice, but my preference is to omit it so we
        don't return a reference that doesn't admit any valid indices.
        """
    m = ConcreteModel()
    m.s1 = Set(initialize=[1, 2, 3])
    m.s2 = Set(initialize=['a', 'b', 'c'])
    m.v = Var(m.s1, m.s2)

    def con_rule(m, i, j):
        if j == 'a':
            return Constraint.Skip
        return m.v[i, j] == 5.0

    def vacuous_con_rule(m, i, j):
        return Constraint.Skip
    m.con = Constraint(m.s1, m.s2, rule=con_rule)
    with self.assertRaises(StopIteration):
        next(iter(m.con[:, 'a']))
    sets = (m.s1,)
    ctype = Constraint
    sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
    self.assertEqual(len(comps_list), 1)
    self.assertEqual(len(comps_list[0]), len(m.s2) - 1)
    m.del_component(m.con)
    m.vacuous_con = Constraint(m.s1, m.s2, rule=vacuous_con_rule)
    with self.assertRaises(StopIteration):
        next(iter(m.vacuous_con[...]))
    sets = (m.s1, m.s2)
    ctype = Constraint
    sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
    self.assertEqual(len(comps_list), 0)
    m.del_component(m.vacuous_con)
    m.del_component(m.v)

    def block_rule(b, i, j):
        b.v = Var()
    m.b = Block(m.s1, m.s2, rule=block_rule)
    for i in m.s1:
        del m.b[i, 'a']
    with self.assertRaises(StopIteration):
        next(iter(m.b[:, 'a'].v))
    sets = (m.s1,)
    ctype = Var
    sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
    self.assertEqual(len(comps_list), 1)
    self.assertEqual(len(comps_list[0]), len(m.s2) - 1)
    for idx in m.b:
        del m.b[idx]
    with self.assertRaises(StopIteration):
        next(iter(m.b[...].v))
    sets = (m.s1, m.s2)
    ctype = Var
    sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
    self.assertEqual(len(comps_list), 0)
    subset_set = ComponentSet(m.b.index_set().subsets())
    for s in sets:
        self.assertIn(s, subset_set)