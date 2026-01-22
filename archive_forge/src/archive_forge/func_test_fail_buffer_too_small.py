import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
def test_fail_buffer_too_small(self, writable_file):
    ref_data = numpy.arange(16).reshape(4, 4)
    dataset = writable_file.create_dataset('uncompressed', data=ref_data, chunks=ref_data.shape)
    out = bytearray(ref_data.nbytes // 2)
    with pytest.raises(ValueError):
        dataset.id.read_direct_chunk((0, 0), out=out)