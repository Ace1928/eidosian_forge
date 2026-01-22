import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_arccosh(self):
    q = [1, 2, 3, 4, 6] * pq.dimensionless
    self.assertQuantityEqual(np.arccosh(q), np.arccosh(q.magnitude) * pq.rad)