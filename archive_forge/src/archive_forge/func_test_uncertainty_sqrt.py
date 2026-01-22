from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_uncertainty_sqrt(self):
    a = UncertainQuantity([1, 2], 'm', [0.1, 0.2])
    self.assertQuantityEqual(a ** 0.5, a.sqrt())