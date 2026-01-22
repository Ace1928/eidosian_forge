import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.set_utils import (
def test_get_indices_of_projection(self):
    m = ConcreteModel()
    m.s1 = Set(initialize=[1, 2, 3])
    m.s2 = Set(initialize=[4, 5, 6])
    m.s3 = Set(initialize=['a', 'b'])
    m.s4 = Set(initialize=['c', 'd'])
    product = m.s1.cross(m.s2, m.s3, m.s4)
    info = get_indices_of_projection(product, m.s2)
    set_except = info['set_except']
    index_getter = info['index_getter']
    predicted_len = len(m.s1) * len(m.s3) * len(m.s4)
    self.assertEqual(len(set_except), predicted_len)
    removed_index = 4
    for idx in m.s1 * m.s3 * m.s4:
        self.assertIn(idx, set_except)
        full_index = index_getter(idx, removed_index)
        self.assertIn(full_index, product)
    sub_prod = m.s2.cross(m.s3, m.s4)
    product = m.s1.cross(sub_prod)
    info = get_indices_of_projection(product, m.s2)
    set_except = info['set_except']
    index_getter = info['index_getter']
    predicted_len = len(m.s1) * len(m.s3) * len(m.s4)
    self.assertEqual(len(set_except), predicted_len)
    removed_index = 4
    for idx in m.s1 * m.s3 * m.s4:
        self.assertIn(idx, set_except)
        full_index = index_getter(idx, removed_index)
        self.assertIn(full_index, product)
    product = m.s1.cross(m.s2, m.s3, m.s4)
    info = get_indices_of_projection(product, m.s2, m.s4)
    set_except = info['set_except']
    index_getter = info['index_getter']
    predicted_len = len(m.s1) * len(m.s3)
    self.assertEqual(len(set_except), predicted_len)
    removed_index = (4, 'd')
    for idx in m.s1 * m.s3:
        self.assertIn(idx, set_except)
        full_index = index_getter(idx, *removed_index)
        self.assertIn(full_index, product)