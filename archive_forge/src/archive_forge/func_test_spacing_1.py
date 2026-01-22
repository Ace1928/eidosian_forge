import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
@xfail(condition=arch32, reason='Known test failure on 32-bit platforms. See links for details: https://github.com/scikit-image/scikit-image/issues/3091 https://github.com/scikit-image/scikit-image/issues/3092')
def test_spacing_1():
    n = 30
    lx, ly, lz = (n, n, n)
    data, _ = make_3d_syntheticdata(lx, ly, lz)
    data_aniso = np.zeros((n, n * 2, n))
    for i, yz in enumerate(data):
        data_aniso[i, :, :] = resize(yz, (n * 2, n), mode='constant', anti_aliasing=False)
    small_l = int(lx // 5)
    labels_aniso = np.zeros_like(data_aniso)
    labels_aniso[lx // 5, ly // 5, lz // 5] = 1
    labels_aniso[lx // 2 + small_l // 4, ly - small_l // 2, lz // 2 - small_l // 4] = 2
    with expected_warnings(['"cg" mode|scipy.sparse.linalg.cg']):
        labels_aniso = random_walker(data_aniso, labels_aniso, mode='cg', spacing=(1.0, 2.0, 1.0))
    assert (labels_aniso[13:17, 26:34, 13:17] == 2).all()
    data_aniso = np.zeros((n, n * 2, n))
    for i in range(data.shape[1]):
        data_aniso[i, :, :] = resize(data[:, 1, :], (n * 2, n), mode='constant', anti_aliasing=False)
    small_l = int(lx // 5)
    labels_aniso2 = np.zeros_like(data_aniso)
    labels_aniso2[lx // 5, ly // 5, lz // 5] = 1
    labels_aniso2[lx - small_l // 2, ly // 2 + small_l // 4, lz // 2 - small_l // 4] = 2
    with expected_warnings(['"cg" mode|scipy.sparse.linalg.cg']):
        labels_aniso2 = random_walker(data_aniso, labels_aniso2, mode='cg', spacing=(2.0, 1.0, 1.0))
    assert (labels_aniso2[26:34, 13:17, 13:17] == 2).all()