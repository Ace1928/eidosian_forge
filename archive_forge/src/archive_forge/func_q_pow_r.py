import operator as op
from functools import partial
import sys
import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase
def q_pow_r(q1, q2):
    return q1 ** q2