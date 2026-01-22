import pickle
import copy
from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase
def test_unitquantity_persistence(self):
    x = pq.m
    y = pickle.loads(pickle.dumps(x))
    self.assertQuantityEqual(x, y)
    x = pq.CompoundUnit('pc/cm**3')
    y = pickle.loads(pickle.dumps(x))
    self.assertQuantityEqual(x, y)