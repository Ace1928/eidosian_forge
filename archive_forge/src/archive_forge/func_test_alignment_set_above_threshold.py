import h5py
from .common import TestCase
def test_alignment_set_above_threshold(self):
    alignment_threshold = 1000
    alignment_interval = 4096
    for shape in [(1033,), (1000,), (1001,)]:
        fname = self.mktemp()
        with h5py.File(fname, 'w', alignment_threshold=alignment_threshold, alignment_interval=alignment_interval) as h5file:
            for i in range(1000):
                dataset = h5file.create_dataset(dataset_name(i), shape, dtype='uint8')
                dataset[...] = i % 256
                assert is_aligned(dataset, offset=alignment_interval)