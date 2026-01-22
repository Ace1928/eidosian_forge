from ..common import ut
import h5py as h5
import numpy as np
def test_ellipsis_end(self):
    dataset = h5.VirtualSource('test', 'test', (20, 30, 30))
    sliced = dataset[0:1, ...]
    self.assertEqual((1,) + dataset.shape[1:], sliced.shape)