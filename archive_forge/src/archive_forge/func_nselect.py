import numpy as np
from .base import product
from .. import h5s, h5r, _selector
@property
def nselect(self):
    """ Number of elements currently selected """
    return self._id.get_select_npoints()