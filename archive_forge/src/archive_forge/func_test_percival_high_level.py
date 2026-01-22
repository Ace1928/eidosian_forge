import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def test_percival_high_level(self):
    outfile = osp.join(self.working_dir, 'percival.h5')
    layout = h5.VirtualLayout(shape=(79, 200, 200), dtype=np.float64)
    for k, filename in enumerate(self.fname):
        dim1 = 19 if k == 3 else 20
        vsource = h5.VirtualSource(filename, 'data', shape=(dim1, 200, 200))
        layout[k:79:4, :, :] = vsource[:, :, :]
    with h5.File(outfile, 'w', libver='latest') as f:
        f.create_virtual_dataset('data', layout, fillvalue=-5)
    foo = np.array(2 * list(range(4)))
    with h5.File(outfile, 'r') as f:
        ds = f['data']
        line = ds[:8, 100, 100]
        self.assertEqual(ds.shape, (79, 200, 200))
        assert_array_equal(line, foo)