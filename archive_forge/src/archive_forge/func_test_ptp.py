from .. import units as pq
from .common import TestCase
import numpy as np
def test_ptp(self):
    self.methodWithOut('ptp', 3 * pq.m)
    self.methodWithOut('ptp', [2, 2] * pq.m, axis=0)
    self.methodWithOut('ptp', [1, 1] * pq.m, axis=1)