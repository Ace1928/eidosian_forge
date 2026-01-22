from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_set_uncertainty(self):
    a = UncertainQuantity([1, 2], 'm', [0.1, 0.2])
    a.uncertainty = [1.0, 2.0] * pq.m
    self.assertQuantityEqual(a.uncertainty, [1, 2] * pq.m)

    def set_u(q, u):
        q.uncertainty = u
    self.assertRaises(ValueError, set_u, a, 1)
    self.assertRaises(ValueError, set_u, a, [1, 2])