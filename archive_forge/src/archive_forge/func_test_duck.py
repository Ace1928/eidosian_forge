import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_duck(self):

    class DummyComplexArray:

        @property
        def dtype(self):
            return np.dtype(complex)
    dummy = DummyComplexArray()
    assert_(iscomplexobj(dummy))