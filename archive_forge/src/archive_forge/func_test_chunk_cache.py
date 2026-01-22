import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
def test_chunk_cache(self):
    """test get/set chunk cache """
    dalist = h5p.create(h5p.DATASET_ACCESS)
    nslots = 10000
    nbytes = 1000000
    w0 = 0.5
    dalist.set_chunk_cache(nslots, nbytes, w0)
    self.assertEqual((nslots, nbytes, w0), dalist.get_chunk_cache())