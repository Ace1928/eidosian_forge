import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def test_inspection(self):
    with h5.File(osp.join(self.tmpdir, '1.h5'), 'r') as f:
        assert not f['data'].is_virtual
    outfile = self.make_virtual_ds()
    with h5.File(outfile, 'r') as f:
        ds = f['/group/data']
        assert ds.is_virtual
        src_files = {osp.join(self.tmpdir, '{}.h5'.format(n)) for n in range(1, 5)}
        assert {s.file_name for s in ds.virtual_sources()} == src_files