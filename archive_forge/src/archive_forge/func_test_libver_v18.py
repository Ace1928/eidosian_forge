import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
def test_libver_v18(self):
    """ Test libver bounds set/get for H5F_LIBVER_V18"""
    plist = h5p.create(h5p.FILE_ACCESS)
    plist.set_libver_bounds(h5f.LIBVER_EARLIEST, h5f.LIBVER_V18)
    self.assertEqual((h5f.LIBVER_EARLIEST, h5f.LIBVER_V18), plist.get_libver_bounds())