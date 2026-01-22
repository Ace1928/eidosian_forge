from .. import units as pq
from .common import TestCase
import numpy as np
def test_nanargmax(self):
    q = np.append(self.q, np.nan) * self.q.units
    self.assertEqual(self.q.nanargmax(), 3)