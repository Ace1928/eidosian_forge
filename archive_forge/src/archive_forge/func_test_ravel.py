from .. import units as pq
from .common import TestCase
import numpy as np
def test_ravel(self):
    self.assertQuantityEqual(self.q.ravel(), [1, 2, 3, 4] * pq.m)