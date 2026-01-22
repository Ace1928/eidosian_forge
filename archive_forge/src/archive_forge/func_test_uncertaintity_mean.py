from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_uncertaintity_mean(self):
    a = UncertainQuantity([1, 2], 'm', [0.1, 0.2])
    mean0 = np.sum(a) / np.size(a)
    mean1 = a.mean()
    self.assertQuantityEqual(mean0, mean1)