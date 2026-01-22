from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_uncertainquantity_divide(self):
    a = UncertainQuantity([1, 2], 'm', [0.1, 0.2])
    self.assertQuantityEqual(a / a, [1.0, 1.0])
    self.assertQuantityEqual((a / a).uncertainty, [0.14142, 0.14142])
    self.assertQuantityEqual(a / pq.m, [1.0, 2.0])
    self.assertQuantityEqual((a / pq.m).uncertainty, [0.1, 0.2])
    self.assertQuantityEqual(a / 2, [0.5, 1.0] * pq.m)
    self.assertQuantityEqual((a / 2).uncertainty, [0.05, 0.1] * pq.m)
    self.assertQuantityEqual(1 / a, [1.0, 0.5] / pq.m)
    self.assertQuantityEqual((1 / a).uncertainty, [0.1, 0.05] / pq.m)