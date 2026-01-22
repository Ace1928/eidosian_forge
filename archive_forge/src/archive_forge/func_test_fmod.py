import operator as op
from functools import partial
import sys
import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase
def test_fmod(self):
    self.assertQuantityEqual(np.fmod(10 * pq.m, 3 * pq.m), 1 * pq.m)
    self.assertRaises(ValueError, np.fmod, 10 * pq.J, 3 * pq.m)