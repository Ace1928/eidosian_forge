import pytest
import numpy as np
from skimage import data
from skimage.measure._label import _label_bool, label
from skimage.measure._ccomp import label_cython as clabel
from skimage._shared import testing
def test_connectivity():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    for c in (1, 2, 3):
        l_ndi = _label_bool(img, connectivity=c)
        l_cy = clabel(img, connectivity=c)
        testing.assert_equal(l_ndi, l_cy)
    for c in (0, 4):
        with pytest.raises(ValueError):
            l_ndi = _label_bool(img, connectivity=c)
        with pytest.raises(ValueError):
            l_cy = clabel(img, connectivity=c)