import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
@testing.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_2d_cg(dtype):
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    data = data.astype(dtype, copy=False)
    with expected_warnings(['"cg" mode|scipy.sparse.linalg.cg']):
        labels_cg = random_walker(data, labels, beta=90, mode='cg')
    assert (labels_cg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    with expected_warnings(['"cg" mode|scipy.sparse.linalg.cg']):
        full_prob = random_walker(data, labels, beta=90, mode='cg', return_full_prob=True)
    assert (full_prob[1, 25:45, 40:60] >= full_prob[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape