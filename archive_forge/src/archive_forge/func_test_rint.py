import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_rint(self):
    a = [-4.1, -3.6, -2.5, 0.1, 2.5, 3.1, 3.9] * pq.m
    self.assertQuantityEqual(np.rint(a), [-4.0, -4.0, -2.0, 0.0, 2.0, 3.0, 4.0] * pq.m)