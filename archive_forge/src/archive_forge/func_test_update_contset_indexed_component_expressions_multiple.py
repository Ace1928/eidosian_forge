import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_contset_indexed_component_expressions_multiple(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.t2 = ContinuousSet(initialize=[1, 2, 3])
    m.s1 = Set(initialize=[1, 2, 3])
    m.s2 = Set(initialize=[(1, 1), (2, 2)])

    def _init(m, i, j):
        return j + i
    m.p1 = Param(m.s1, m.t, default=_init)
    m.v1 = Var(m.s1, m.t, initialize=5)
    m.v2 = Var(m.s2, m.t, initialize=2)
    m.v3 = Var(m.t2, m.s2, initialize=1)

    def _con1(m, si, ti):
        return m.v1[si, ti] * m.p1[si, ti]
    m.con1 = Expression(m.s1, m.t, rule=_con1)

    def _con2(m, i, j, ti):
        return m.v2[i, j, ti] + m.p1[1, ti]
    m.con2 = Expression(m.s2, m.t, rule=_con2)

    def _con3(m, i, ti, ti2, j, k):
        return m.v1[i, ti] - m.v3[ti2, j, k] * m.p1[i, ti]
    m.con3 = Expression(m.s1, m.t, m.t2, m.s2, rule=_con3)
    expansion_map = ComponentMap()
    generate_finite_elements(m.t, 5)
    update_contset_indexed_component(m.p1, expansion_map)
    update_contset_indexed_component(m.v1, expansion_map)
    update_contset_indexed_component(m.v2, expansion_map)
    update_contset_indexed_component(m.v3, expansion_map)
    update_contset_indexed_component(m.con1, expansion_map)
    update_contset_indexed_component(m.con2, expansion_map)
    update_contset_indexed_component(m.con3, expansion_map)
    self.assertTrue(len(m.con1) == 18)
    self.assertTrue(len(m.con2) == 12)
    self.assertTrue(len(m.con3) == 108)
    self.assertEqual(m.con1[1, 4](), 25)
    self.assertEqual(m.con1[2, 6](), 40)
    self.assertEqual(m.con1[3, 8](), 55)
    self.assertEqual(m.con2[1, 1, 2](), 5)
    self.assertEqual(m.con2[2, 2, 4](), 7)
    self.assertEqual(m.con2[1, 1, 8](), 11)
    self.assertEqual(m.con3[1, 2, 1, 1, 1](), 2)
    self.assertEqual(m.con3[1, 4, 1, 2, 2](), 0)
    self.assertEqual(m.con3[2, 6, 3, 1, 1](), -3)
    self.assertEqual(m.con3[3, 8, 2, 2, 2](), -6)