import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_contset_indexed_component_block_multiple(self):
    model = ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 10))
    model.s1 = Set(initialize=['A', 'B', 'C'])
    model.s2 = Set(initialize=[('x1', 'x1'), ('x2', 'x2')])

    def _block_rule(b, t, s1):
        m = b.model()

        def _init(m, i, j):
            return j * 2
        b.p1 = Param(m.s1, m.t, mutable=True, default=_init)
        b.v1 = Var(m.s1, m.t, initialize=5)
        b.v2 = Var(m.s2, m.t, initialize=2)
        b.v3 = Var(m.t, m.s2, initialize=1)

        def _con1(_b, si, ti):
            return _b.v1[si, ti] * _b.p1[si, ti] == _b.v1[si, t] ** 2
        b.con1 = Constraint(m.s1, m.t, rule=_con1)

        def _con2(_b, i, j, ti):
            return _b.v2[i, j, ti] - _b.v3[ti, i, j] + _b.p1['A', ti]
        b.con2 = Expression(m.s2, m.t, rule=_con2)
    model.blk = Block(model.t, model.s1, rule=_block_rule)
    expansion_map = ComponentMap()
    self.assertTrue(len(model.blk), 6)
    generate_finite_elements(model.t, 5)
    missing_idx = set(model.blk.index_set()) - set(model.blk._data.keys())
    model.blk._dae_missing_idx = missing_idx
    update_contset_indexed_component(model.blk, expansion_map)
    self.assertEqual(len(model.blk), 18)
    self.assertEqual(len(model.blk[10, 'C'].con1), 6)
    self.assertEqual(len(model.blk[2, 'B'].con1), 18)
    self.assertEqual(len(model.blk[10, 'C'].v2), 4)
    self.assertEqual(model.blk[2, 'A'].p1['A', 2].value, 4)
    self.assertEqual(model.blk[8, 'C'].p1['B', 6].value, 12)
    self.assertEqual(model.blk[4, 'B'].con1['B', 4](), 15)
    self.assertEqual(model.blk[6, 'A'].con1['C', 8](), 55)
    self.assertEqual(model.blk[0, 'A'].con2['x1', 'x1', 10](), 21)
    self.assertEqual(model.blk[4, 'C'].con2['x2', 'x2', 6](), 13)