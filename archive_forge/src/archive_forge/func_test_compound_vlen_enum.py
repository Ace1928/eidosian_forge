from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_compound_vlen_enum(self):
    eidt = h5py.enum_dtype({'OFF': 0, 'ON': 1}, basetype=np.uint8)
    vidt = h5py.vlen_dtype(np.uint8)

    def a(items):
        return np.array(items, dtype=np.uint8)
    f = self.f
    dt_vve = np.dtype([('foo', vidt), ('bar', vidt), ('switch', eidt)])
    vve = f.create_dataset('dt_vve', shape=(2,), dtype=dt_vve)
    data = np.array([(a([1, 2, 3]), a([1, 2]), 1), (a([]), a([2, 4, 6]), 0)], dtype=dt_vve)
    vve[:] = data
    actual = vve[:]
    self.assertVlenArrayEqual(data['foo'], actual['foo'])
    self.assertVlenArrayEqual(data['bar'], actual['bar'])
    self.assertArrayEqual(data['switch'], actual['switch'])