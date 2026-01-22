import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_less_equal(self):
    arr1 = (1, 1) * pq.m
    arr2 = (1.0, 2.0) * pq.m
    self.assertTrue(np.all(np.less_equal(arr1, arr2)))
    self.assertFalse(np.all(np.less_equal(arr2, arr1)))