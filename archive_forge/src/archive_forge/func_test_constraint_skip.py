import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_constraint_skip(self):
    m = ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 1))
    m.v = Var(m.time)

    def c_rule(m, t):
        if t == m.time.first():
            return Constraint.Skip
        return m.v[t] == 1.0
    m.c = Constraint(m.time, rule=c_rule)
    scalar, dae = flatten_dae_components(m, m.time, Constraint)
    ref_data = {self._hashRef(Reference(m.c[:]))}
    self.assertEqual(len(dae), len(ref_data))
    for ref in dae:
        self.assertIn(self._hashRef(ref), ref_data)