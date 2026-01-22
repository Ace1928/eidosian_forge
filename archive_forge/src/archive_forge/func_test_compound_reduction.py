import unittest
import numpy as np
from .. import units as pq
from .. import quantity
from .common import TestCase
def test_compound_reduction(self):
    pc_per_cc = pq.CompoundUnit('pc/cm**3')
    temp = pc_per_cc * pq.CompoundUnit('m/m**3')
    self.assertQuantityEqual(temp.simplified, 3.08568025e+22 / pq.m ** 4, delta=1e+17)
    self.assertQuantityEqual(temp.rescale('pc**-4'), 2.79740021556e+88 / pq.pc ** 4, delta=1e+83)