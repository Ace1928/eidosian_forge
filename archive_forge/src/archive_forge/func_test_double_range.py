from ..common import ut
import h5py as h5
import numpy as np
def test_double_range(self):
    dataset = h5.VirtualSource('test', 'test', (20, 30, 30))
    sliced = dataset[5:10, :, 20:25]
    self.assertEqual((5, 30, 5), sliced.shape)