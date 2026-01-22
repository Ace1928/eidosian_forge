from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def test_uncertaintity_nanmean(self):
    a = UncertainQuantity([1, 2], 'm', [0.1, 0.2])
    b = UncertainQuantity([1, 2, np.nan], 'm', [0.1, 0.2, np.nan])
    self.assertQuantityEqual(a.mean(), b.nanmean())