from ..common import ut
import numpy as np
import h5py as h5
import tempfile
def test_virtual_prefix(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    src_file = h5.File(tmp_path / 'a' / 'src.h5', 'w')
    src_file['data'] = np.arange(10)
    vds_file = h5.File(tmp_path / 'b' / 'vds.h5', 'w')
    layout = h5.VirtualLayout(shape=(10,), dtype=np.int64)
    layout[:] = h5.VirtualSource('src.h5', 'data', shape=(10,))
    vds_file.create_virtual_dataset('data', layout, fillvalue=-1)
    np.testing.assert_array_equal(vds_file['data'], np.full(10, fill_value=-1))
    path_a = bytes(tmp_path / 'a')
    dapl = h5.h5p.create(h5.h5p.DATASET_ACCESS)
    dapl.set_virtual_prefix(path_a)
    vds_id = h5.h5d.open(vds_file.id, b'data', dapl=dapl)
    vds = h5.Dataset(vds_id)
    np.testing.assert_array_equal(vds[:], np.arange(10))
    assert vds.id.get_access_plist().get_virtual_prefix() == path_a