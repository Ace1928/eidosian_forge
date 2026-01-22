import copy
from functools import wraps
import numpy as np
from . import markup
from .dimensionality import Dimensionality, p_dict
from .registry import unit_registry
from .decorators import with_doc
def scale_other_units(f):

    @wraps(f)
    def g(self, other, *args):
        other = np.asanyarray(other)
        if not isinstance(other, Quantity):
            other = other.view(type=Quantity)
        if other._dimensionality != self._dimensionality:
            other = other.rescale(self.units, dtype=np.result_type(self.dtype, other.dtype))
        return f(self, other, *args)
    return g