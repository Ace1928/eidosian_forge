import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
def test_fail_buffer_readonly(self, writable_file):
    ref_data = numpy.arange(16).reshape(4, 4)
    dataset = writable_file.create_dataset('uncompressed', data=ref_data, chunks=ref_data.shape)
    out = bytes(ref_data.nbytes)
    with pytest.raises(BufferError):
        dataset.id.read_direct_chunk((0, 0), out=out)