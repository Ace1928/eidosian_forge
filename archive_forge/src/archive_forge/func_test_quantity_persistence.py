import pickle
import copy
from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase
def test_quantity_persistence(self):
    x = 20 * pq.m
    y = pickle.loads(pickle.dumps(x))
    self.assertQuantityEqual(x, y)