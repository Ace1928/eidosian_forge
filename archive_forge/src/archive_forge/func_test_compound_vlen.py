from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_compound_vlen(self):
    vidt = h5py.vlen_dtype(np.uint8)
    eidt = h5py.enum_dtype({'OFF': 0, 'ON': 1}, basetype=np.uint8)
    for np_align in (False, True):
        dt = np.dtype([('a', eidt), ('foo', vidt), ('bar', vidt), ('switch', eidt)], align=np_align)
        np_offsets = [dt.fields[i][1] for i in dt.names]
        for logical in (False, True):
            if logical and np_align:
                self.assertRaises(TypeError, h5py.h5t.py_create, dt, logical=logical)
            else:
                ht = h5py.h5t.py_create(dt, logical=logical)
                offsets = [ht.get_member_offset(i) for i in range(ht.get_nmembers())]
                if np_align:
                    self.assertEqual(np_offsets, offsets)