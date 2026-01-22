import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_reshape_infer_two_unknowns(self):
    nparr = np.empty((3, 4, 5))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    with self.assertRaises(ValueError) as raises:
        arr.reshape(-1, -1, 3)
    self.assertIn('can only specify one unknown dimension', str(raises.exception))