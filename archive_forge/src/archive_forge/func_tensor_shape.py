from __future__ import annotations
import copy
from functools import reduce
from operator import mul
from math import log2
from numbers import Integral
from qiskit.exceptions import QiskitError
@property
def tensor_shape(self):
    """Return a tuple of the tensor shape"""
    return tuple(reversed(self.dims_l())) + tuple(reversed(self.dims_r()))