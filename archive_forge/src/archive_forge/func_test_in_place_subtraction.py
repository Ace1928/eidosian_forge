import operator as op
from functools import partial
import sys
import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase
def test_in_place_subtraction(self):
    x = 1 * pq.m
    x -= pq.m
    self.assertQuantityEqual(x, 0 * pq.m)
    x = 1 * pq.m
    x -= -pq.m
    self.assertQuantityEqual(x, 2 * pq.m)
    x = [1, 2, 3, 4] * pq.m
    x -= pq.m
    self.assertQuantityEqual(x, [0, 1, 2, 3] * pq.m)
    x = [1, 2, 3, 4] * pq.m
    x -= [1, 1, 1, 1] * pq.m
    self.assertQuantityEqual(x, [0, 1, 2, 3] * pq.m)
    x = [1, 2, 3, 4] * pq.m
    x[:2] -= pq.m
    self.assertQuantityEqual(x, [0, 1, 3, 4] * pq.m)
    x = [1, 2, 3, 4] * pq.m
    x[:2] -= -pq.m
    self.assertQuantityEqual(x, [2, 3, 3, 4] * pq.m)
    x = [1, 2, 3, 4] * pq.m
    x[:2] -= [1, 2] * pq.m
    self.assertQuantityEqual(x, [0, 0, 3, 4] * pq.m)
    x = [1, 2, 3, 4] * pq.m
    x[::2] -= [1, 2] * pq.m
    self.assertQuantityEqual(x, [0, 2, 1, 4] * pq.m)
    self.assertRaises(ValueError, op.isub, 1 * pq.m, 1)
    self.assertRaises(ValueError, op.isub, 1 * pq.m, pq.J)
    self.assertRaises(ValueError, op.isub, 1 * pq.m, 5 * pq.J)
    self.assertRaises(ValueError, op.isub, [1, 2, 3] * pq.m, 1)
    self.assertRaises(ValueError, op.isub, [1, 2, 3] * pq.m, pq.J)
    self.assertRaises(ValueError, op.isub, [1, 2, 3] * pq.m, 5 * pq.J)