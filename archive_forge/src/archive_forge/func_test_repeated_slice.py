from ..common import ut
import h5py as h5
import numpy as np
def test_repeated_slice(self):
    dataset = h5.VirtualSource('test', 'test', (20, 30, 30))
    sliced = dataset[5:10, :, :]
    with self.assertRaises(RuntimeError):
        sliced[:, :4]