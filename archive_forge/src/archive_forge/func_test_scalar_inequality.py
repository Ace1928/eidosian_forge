import operator as op
from .. import units as pq
from .common import TestCase
def test_scalar_inequality(self):
    self.assertEqual(pq.J != pq.erg, [True])
    self.assertEqual(2 * pq.J != pq.J, [True])
    self.assertEqual(str(2 * pq.J) != str(pq.J), True)
    self.assertEqual(pq.J != 2 * pq.kg * pq.m ** 2 / pq.s ** 2, [True])
    self.assertEqual(pq.J != pq.J, [False])
    self.assertEqual(1 * pq.J != pq.J, [False])
    self.assertEqual(pq.J != 1 * pq.kg * pq.m ** 2 / pq.s ** 2, [False])