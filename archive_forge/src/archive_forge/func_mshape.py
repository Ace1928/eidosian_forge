import numpy as np
from .base import product
from .. import h5s, h5r, _selector
@property
def mshape(self):
    return self._mshape