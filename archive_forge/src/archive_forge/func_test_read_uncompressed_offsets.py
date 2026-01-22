import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
def test_read_uncompressed_offsets(self):
    filename = self.mktemp().encode()
    frame = numpy.arange(16).reshape(4, 4)
    with h5py.File(filename, 'w') as filehandle:
        dataset = filehandle.create_dataset('frame', maxshape=(1,) + frame.shape, shape=(1,) + frame.shape, compression='gzip', compression_opts=9)
        DISABLE_ALL_FILTERS = 4294967295
        dataset.id.write_direct_chunk((0, 0, 0), frame.tobytes(), filter_mask=DISABLE_ALL_FILTERS)
    with h5py.File(filename, 'r') as filehandle:
        dataset = filehandle['frame']
        filter_mask, compressed_frame = dataset.id.read_direct_chunk((0, 0, 0))
    self.assertNotEqual(filter_mask, 0)
    self.assertEqual(compressed_frame, frame.tobytes())