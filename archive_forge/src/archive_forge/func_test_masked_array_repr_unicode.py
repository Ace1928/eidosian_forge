import numpy as np
from numpy.testing import (
def test_masked_array_repr_unicode(self):
    repr(np.ma.array('Unicode'))