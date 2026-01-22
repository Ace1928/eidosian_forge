import numpy as np
from . import _zeros_py as optzeros
from ._numdiff import approx_derivative
def ncalls(self):
    return self.n_calls