from ..common import ut
import h5py as h5
import numpy as np
def test_shape_calculation_positive_step(self):
    dataset = h5.VirtualSource('test', 'test', (20,))
    cmp = []
    for i in range(5):
        d = dataset[2:12 + i:3].shape[0]
        ref = np.arange(20)[2:12 + i:3].size
        cmp.append(ref == d)
    self.assertEqual(5, sum(cmp))