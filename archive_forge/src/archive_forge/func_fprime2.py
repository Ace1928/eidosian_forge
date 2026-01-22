import numpy as np
from . import _zeros_py as optzeros
from ._numdiff import approx_derivative
def fprime2(self, x, *args):
    """Calculate f'' or use a cached value if available"""
    if self.vals is None or x != self.x:
        self(x, *args)
    return self.vals[2]