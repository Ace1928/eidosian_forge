from ..common import ut
import h5py as h5
import numpy as np
def test_full_slice(self):
    dataset = h5.VirtualSource('test', 'test', (20, 30, 30))
    sliced = dataset[:, :, :]
    self.assertEqual(dataset.shape, sliced.shape)