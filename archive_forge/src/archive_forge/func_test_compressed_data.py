import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
@pytest.mark.skipif(h5py.version.hdf5_version_tuple < (1, 10, 5), reason='chunk info requires HDF5 >= 1.10.5')
@pytest.mark.skipif('gzip' not in h5py.filters.encode, reason='DEFLATE is not installed')
def test_compressed_data(self, writable_file):
    ref_data = numpy.arange(16).reshape(4, 4)
    dataset = writable_file.create_dataset('gzip', data=ref_data, chunks=ref_data.shape, compression='gzip', compression_opts=9)
    chunk_info = dataset.id.get_chunk_info(0)
    out = bytearray(chunk_info.size)
    filter_mask, chunk = dataset.id.read_direct_chunk(chunk_info.chunk_offset, out=out)
    assert filter_mask == chunk_info.filter_mask
    assert len(chunk) == chunk_info.size
    assert out == dataset.id.read_direct_chunk(chunk_info.chunk_offset)[1]