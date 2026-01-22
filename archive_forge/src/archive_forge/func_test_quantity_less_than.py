import operator as op
from .. import units as pq
from .common import TestCase
def test_quantity_less_than(self):
    self.assertQuantityEqual([1, 2, 33] * pq.J < [1, 22, 3] * pq.J, [0, 1, 0])
    self.assertQuantityEqual([50, 100, 150] * pq.cm < [1, 1, 1] * pq.m, [1, 0, 0])
    self.assertQuantityEqual([1, 2, 33] * pq.J < [1, 22, 3], [0, 1, 0])
    self.assertRaises(ValueError, op.lt, [1, 2, 33] * pq.J, [1, 22, 3] * pq.kg)