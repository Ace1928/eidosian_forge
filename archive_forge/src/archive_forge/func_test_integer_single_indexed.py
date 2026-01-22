from ..common import ut
import h5py as h5
import numpy as np
def test_integer_single_indexed(self):
    dataset = h5.VirtualSource('test', 'test', (20, 30, 30))
    sliced = dataset[5]
    self.assertEqual((30, 30), sliced.shape)