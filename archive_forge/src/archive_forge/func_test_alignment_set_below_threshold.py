import h5py
from .common import TestCase
def test_alignment_set_below_threshold(self):
    alignment_threshold = 1000
    alignment_interval = 1024
    for shape in [(881,), (999,)]:
        fname = self.mktemp()
        with h5py.File(fname, 'w', alignment_threshold=alignment_threshold, alignment_interval=alignment_interval) as h5file:
            for i in range(1000):
                dataset = h5file.create_dataset(dataset_name(i), shape, dtype='uint8')
                dataset[...] = i
                if not is_aligned(dataset, offset=alignment_interval):
                    break
            else:
                raise RuntimeError(f'Data was all found to be aligned to {alignment_interval}. This is highly unlikely.')