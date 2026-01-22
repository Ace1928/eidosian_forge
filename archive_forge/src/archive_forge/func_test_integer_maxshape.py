from ..common import ut
import h5py as h5
import numpy as np
def test_integer_maxshape(self):
    dataset = h5.VirtualSource('test', 'test', 20, maxshape=30)
    self.assertEqual(dataset.maxshape, (30,))