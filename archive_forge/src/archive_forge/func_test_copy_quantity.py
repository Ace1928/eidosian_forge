import pickle
import copy
from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase
def test_copy_quantity(self):
    for dtype in [float, object]:
        x = (20 * pq.m).astype(dtype)
        y = copy.copy(x)
        self.assertQuantityEqual(x, y)