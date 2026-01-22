import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
@ut.skipIf(version.hdf5_version_tuple < (1, 11, 4), 'Requires HDF5 1.11.4 or later')
def test_libver_v112(self):
    """ Test libver bounds set/get for H5F_LIBVER_V112"""
    plist = h5p.create(h5p.FILE_ACCESS)
    plist.set_libver_bounds(h5f.LIBVER_V18, h5f.LIBVER_V112)
    self.assertEqual((h5f.LIBVER_V18, h5f.LIBVER_V112), plist.get_libver_bounds())