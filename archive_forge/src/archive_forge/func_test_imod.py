import operator as op
from functools import partial
import sys
import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase
def test_imod(self):
    x = 10 * pq.m
    x %= 3 * pq.m
    self.assertQuantityEqual(x, 1 * pq.m)
    x = 10 * pq.m
    x %= (3 * pq.m).rescale('ft')
    self.assertQuantityEqual(x, 10 * pq.m % (3 * pq.m))
    self.assertRaises(ValueError, lambda: op.imod(10 * pq.J, 3 * pq.m))