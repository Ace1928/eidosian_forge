import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def create_excalibur_fem_stripe_datafile(self, fname, nframes, excalibur_data, scale):
    shape = (nframes,) + excalibur_data.fem_stripe_dimensions
    max_shape = shape
    chunk = (1,) + excalibur_data.fem_stripe_dimensions
    with h5.File(fname, 'w', libver='latest') as f:
        dset = f.create_dataset('data', shape=shape, maxshape=max_shape, chunks=chunk, dtype='uint16')
        for data_value_index in np.arange(nframes):
            dset[data_value_index] = excalibur_data.generate_fem_stripe_image(data_value_index * scale)