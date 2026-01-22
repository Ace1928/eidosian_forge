import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_discretized_params_multiple(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.s1 = Set(initialize=[1, 2, 3])
    m.s2 = Set(initialize=[(1, 1), (2, 2)])

    def _rule1(m, i):
        return i ** 2
    m.p1 = Param(m.s1, m.t, initialize=2, default=_rule1)
    m.p2 = Param(m.t, m.s1, default=5)

    def _rule2(m, i, j):
        return i + j
    m.p3 = Param(m.s1, m.t, initialize=2, default=_rule2)

    def _rule3(m, i, j, k):
        return i + j + k
    m.p4 = Param(m.s2, m.t, default=_rule3)
    generate_finite_elements(m.t, 5)
    with self.assertRaises(TypeError):
        for i in m.p1:
            m.p1[i]
    for i in m.p2:
        self.assertEqual(m.p2[i], 5)
    for i in m.t:
        for j in m.s1:
            if i == 0 or i == 10:
                self.assertEqual(m.p3[j, i], 2)
            else:
                self.assertEqual(m.p3[j, i], i + j)
    for i in m.t:
        for j in m.s2:
            self.assertEqual(m.p4[j, i], sum(j, i))