from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_aligned_offsets(self):
    dt = np.dtype('i4,i8,i2', align=True)
    ht = h5py.h5t.py_create(dt)
    self.assertEqual(dt.itemsize, ht.get_size())
    self.assertEqual([dt.fields[i][1] for i in dt.names], [ht.get_member_offset(i) for i in range(ht.get_nmembers())])