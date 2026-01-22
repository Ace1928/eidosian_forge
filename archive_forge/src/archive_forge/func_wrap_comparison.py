import copy
from functools import wraps
import numpy as np
from . import markup
from .dimensionality import Dimensionality, p_dict
from .registry import unit_registry
from .decorators import with_doc
def wrap_comparison(f):

    @wraps(f)
    def g(self, other):
        if isinstance(other, Quantity):
            if other._dimensionality != self._dimensionality:
                other = other.rescale(self._dimensionality)
            other = other.magnitude
        return f(self, other)
    return g