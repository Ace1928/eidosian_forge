import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
@testing.parametrize('channel_axis', [0, 1, -1])
@testing.parametrize('dtype', [np.float32, np.float64])
def test_multispectral_2d(dtype, channel_axis):
    lx, ly = (70, 100)
    data, labels = make_2d_syntheticdata(lx, ly)
    data = data.astype(dtype, copy=False)
    data = data[..., np.newaxis].repeat(2, axis=-1)
    data = np.moveaxis(data, -1, channel_axis)
    with expected_warnings(['"cg" mode|scipy.sparse.linalg.cg', 'The probability range is outside']):
        multi_labels = random_walker(data, labels, mode='cg', channel_axis=channel_axis)
    data = np.moveaxis(data, channel_axis, -1)
    assert data[..., 0].shape == labels.shape
    with expected_warnings(['"cg" mode|scipy.sparse.linalg.cg']):
        random_walker(data[..., 0], labels, mode='cg')
    assert (multi_labels.reshape(labels.shape)[25:45, 40:60] == 2).all()
    assert data[..., 0].shape == labels.shape