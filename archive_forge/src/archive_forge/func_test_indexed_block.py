import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_indexed_block(self):
    m = ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 1))
    m.comp = Set(initialize=['a', 'b'])

    def bb_rule(bb, t):
        bb.dae_var = Var()

    def b_rule(b, c):
        b.bb = Block(m.time, rule=bb_rule)
    m.b = Block(m.comp, rule=b_rule)
    scalar, dae = flatten_dae_components(m, m.time, Var)
    self.assertEqual(len(scalar), 0)
    ref_data = {self._hashRef(Reference(m.b['a'].bb[:].dae_var)), self._hashRef(Reference(m.b['b'].bb[:].dae_var))}
    self.assertEqual(len(dae), len(ref_data))
    for ref in dae:
        self.assertIn(self._hashRef(ref), ref_data)