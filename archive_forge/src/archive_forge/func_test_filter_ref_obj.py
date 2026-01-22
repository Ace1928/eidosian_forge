import os
import numpy as np
import h5py
from .common import ut, TestCase
@ut.skipIf('gzip' not in h5py.filters.encode, 'DEFLATE is not installed')
def test_filter_ref_obj(writable_file):
    gzip8 = h5py.filters.Gzip(level=8)
    assert dict(**gzip8) == {'compression': h5py.h5z.FILTER_DEFLATE, 'compression_opts': (8,)}
    ds = writable_file.create_dataset('x', shape=(100,), dtype=np.uint32, compression=gzip8)
    assert ds.compression == 'gzip'
    assert ds.compression_opts == 8