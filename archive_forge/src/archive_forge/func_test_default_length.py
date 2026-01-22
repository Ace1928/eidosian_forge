import unittest
import numpy as np
from .. import units as pq
from .. import quantity
from .common import TestCase
def test_default_length(self):
    pq.set_default_units(length='mm')
    self.assertQuantityEqual(pq.m.simplified, 1000 * pq.mm)
    pq.set_default_units(length='m')
    self.assertQuantityEqual(pq.m.simplified, pq.m)
    self.assertQuantityEqual(pq.mm.simplified, 0.001 * pq.m)