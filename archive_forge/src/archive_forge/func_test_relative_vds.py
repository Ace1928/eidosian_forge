import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def test_relative_vds(self):
    with h5.File(self.f2) as f:
        data = f['virtual'][:]
        np.testing.assert_array_equal(data[0], self.data1)
        np.testing.assert_array_equal(data[1], self.data2)
    f3 = osp.join(self.tmpdir, 'testfile3.h5')
    os.rename(self.f2, f3)
    with h5.File(f3) as f:
        data = f['virtual'][:]
        assert data.dtype == 'f4'
        np.testing.assert_array_equal(data[0], self.data1)
        np.testing.assert_array_equal(data[1], self.data2)
    f4 = osp.join(self.tmpdir, 'testfile4.h5')
    os.rename(self.f1, f4)
    with h5.File(f3) as f:
        data = f['virtual'][:]
        assert data.dtype == 'f4'
        np.testing.assert_array_equal(data[0], 0)
        np.testing.assert_array_equal(data[1], self.data2)