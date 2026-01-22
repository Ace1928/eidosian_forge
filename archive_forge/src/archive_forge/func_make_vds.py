import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def make_vds(self, f):
    with f.build_virtual_dataset('virtual', (2, 10), dtype='f4') as layout:
        layout[0] = h5.VirtualSource(self.f1, 'data', shape=(10,))
        layout[1] = h5.VirtualSource(self.f2, 'data', shape=(10,))