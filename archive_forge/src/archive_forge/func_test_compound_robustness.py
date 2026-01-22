from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_compound_robustness(self):
    fields = [('f0', np.float64, 25), ('f1', np.uint64, 9), ('f2', np.uint32, 0), ('f3', np.uint16, 5)]
    lastfield = fields[np.argmax([x[2] for x in fields])]
    itemsize = lastfield[2] + np.dtype(lastfield[1]).itemsize + 6
    extract_index = lambda index, sequence: [x[index] for x in sequence]
    dt = np.dtype({'names': extract_index(0, fields), 'formats': extract_index(1, fields), 'offsets': extract_index(2, fields), 'itemsize': itemsize})
    self.assertTrue(dt.itemsize == itemsize)
    data = np.zeros(10, dtype=dt)
    f1 = np.array([1 + i * 4 for i in range(data.shape[0])], dtype=dt.fields['f1'][0])
    f2 = np.array([2 + i * 4 for i in range(data.shape[0])], dtype=dt.fields['f2'][0])
    f3 = np.array([3 + i * 4 for i in range(data.shape[0])], dtype=dt.fields['f3'][0])
    f0c = 3.14
    data['f0'] = f0c
    data['f3'] = f3
    data['f1'] = f1
    data['f2'] = f2
    self.assertTrue(np.all(data['f0'] == f0c))
    self.assertArrayEqual(data['f3'], f3)
    self.assertArrayEqual(data['f1'], f1)
    self.assertArrayEqual(data['f2'], f2)
    fname = self.mktemp()
    with h5py.File(fname, 'w') as fd:
        fd.create_dataset('data', data=data)
    with h5py.File(fname, 'r') as fd:
        readback = fd['data']
        self.assertTrue(readback.dtype == dt)
        self.assertArrayEqual(readback, data)
        self.assertTrue(np.all(readback['f0'] == f0c))
        self.assertArrayEqual(readback['f1'], f1)
        self.assertArrayEqual(readback['f2'], f2)
        self.assertArrayEqual(readback['f3'], f3)