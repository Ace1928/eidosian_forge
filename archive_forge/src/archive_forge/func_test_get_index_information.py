import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_get_index_information(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.x = ContinuousSet(bounds=(0, 10))
    m.s = Set(initialize=['a', 'b', 'c'])
    m.v = Var(m.t, m.x, m.s, initialize=1)
    m.v2 = Var(m.t, m.s, initialize=1)
    disc = TransformationFactory('dae.collocation')
    disc.apply_to(m, wrt=m.t, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
    disc.apply_to(m, wrt=m.x, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
    info = get_index_information(m.v, m.t)
    nts = info['non_ds']
    index_getter = info['index function']
    self.assertEqual(len(nts), 33)
    self.assertTrue(m.x in nts.set_tuple)
    self.assertTrue(m.s in nts.set_tuple)
    self.assertEqual(index_getter((8.0, 'a'), 1, 0), (2.0, 8.0, 'a'))
    info = get_index_information(m.v2, m.t)
    nts = info['non_ds']
    index_getter = info['index function']
    self.assertEqual(len(nts), 3)
    self.assertTrue(m.s is nts)
    self.assertEqual(index_getter('a', 1, 0), (2.0, 'a'))