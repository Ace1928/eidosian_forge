import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
@testing.parametrize('dtype', [np.float32, np.float64])
def test_multispectral_3d(dtype):
    n = 30
    lx, ly, lz = (n, n, n)
    data, labels = make_3d_syntheticdata(lx, ly, lz)
    data = data.astype(dtype, copy=False)
    data = data[..., np.newaxis].repeat(2, axis=-1)
    with expected_warnings(['"cg" mode|scipy.sparse.linalg.cg']):
        multi_labels = random_walker(data, labels, mode='cg', channel_axis=-1)
    assert data[..., 0].shape == labels.shape
    with expected_warnings(['"cg" mode|scipy.sparse.linalg.cg']):
        single_labels = random_walker(data[..., 0], labels, mode='cg')
    assert (multi_labels.reshape(labels.shape)[13:17, 13:17, 13:17] == 2).all()
    assert (single_labels.reshape(labels.shape)[13:17, 13:17, 13:17] == 2).all()
    assert data[..., 0].shape == labels.shape