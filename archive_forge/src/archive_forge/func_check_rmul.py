import operator as op
from functools import partial
import sys
import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase
def check_rmul(self, m1, m2):
    self.assertQuantityEqual(m1 * pq.m, Quantity(m1, 'm'))
    q2 = Quantity(m2, 's')
    a1 = np.asarray(m1)
    a2 = np.asarray(m2)
    self.assertQuantityEqual(m1 * q2, Quantity(a1 * a2, 's'))