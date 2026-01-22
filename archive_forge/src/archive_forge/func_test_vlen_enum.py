from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_vlen_enum(self):
    fname = self.mktemp()
    arr1 = [[1], [1, 2]]
    dt1 = h5py.vlen_dtype(h5py.enum_dtype(dict(foo=1, bar=2), 'i'))
    with h5py.File(fname, 'w') as f:
        df1 = f.create_dataset('test', (len(arr1),), dtype=dt1)
        df1[:] = np.array(arr1, dtype=object)
    with h5py.File(fname, 'r') as f:
        df2 = f['test']
        dt2 = df2.dtype
        arr2 = [e.tolist() for e in df2[:]]
    self.assertEqual(arr1, arr2)
    self.assertEqual(h5py.check_enum_dtype(h5py.check_vlen_dtype(dt1)), h5py.check_enum_dtype(h5py.check_vlen_dtype(dt2)))