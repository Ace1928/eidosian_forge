import operator as op
from .. import units as pq
from .common import TestCase
def test_scalar_comparison(self):
    self.assertEqual(2 * pq.J > pq.J, [True])
    self.assertEqual(2 * pq.J > 1 * pq.J, [True])
    self.assertEqual(1 * pq.J >= pq.J, [True])
    self.assertEqual(1 * pq.J >= 1 * pq.J, [True])
    self.assertEqual(2 * pq.J >= pq.J, [True])
    self.assertEqual(2 * pq.J >= 1 * pq.J, [True])
    self.assertEqual(0.5 * pq.J < pq.J, [True])
    self.assertEqual(0.5 * pq.J < 1 * pq.J, [True])
    self.assertEqual(0.5 * pq.J <= pq.J, [True])
    self.assertEqual(0.5 * pq.J <= 1 * pq.J, [True])
    self.assertEqual(1.0 * pq.J <= pq.J, [True])
    self.assertEqual(1.0 * pq.J <= 1 * pq.J, [True])
    self.assertEqual(2 * pq.J < pq.J, [False])
    self.assertEqual(2 * pq.J < 1 * pq.J, [False])
    self.assertEqual(2 * pq.J <= pq.J, [False])
    self.assertEqual(2 * pq.J <= 1 * pq.J, [False])
    self.assertEqual(0.5 * pq.J > pq.J, [False])
    self.assertEqual(0.5 * pq.J > 1 * pq.J, [False])
    self.assertEqual(0.5 * pq.J >= pq.J, [False])
    self.assertEqual(0.5 * pq.J >= 1 * pq.J, [False])