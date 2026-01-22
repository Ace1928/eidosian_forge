from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_opaque(writable_file):
    arr = np.zeros(3, dtype='V2')
    ds = writable_file.create_dataset('v', data=arr)
    assert isinstance(ds.id.get_type(), h5py.h5t.TypeOpaqueID)
    assert ds.id.get_type().get_size() == 2
    np.testing.assert_array_equal(ds[:], arr)