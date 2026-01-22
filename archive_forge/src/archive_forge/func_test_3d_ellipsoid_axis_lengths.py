import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def test_3d_ellipsoid_axis_lengths():
    """Verify that estimated axis lengths are correct.

    Uses an ellipsoid at an arbitrary position and orientation.
    """
    half_lengths = (20, 10, 50)
    e = draw.ellipsoid(*half_lengths).astype(int)
    e = np.pad(e, pad_width=[(30, 18), (30, 12), (40, 20)], mode='constant')
    R = transform.EuclideanTransform(rotation=[0.2, 0.3, 0.4], dimensionality=3)
    e = ndi.affine_transform(e, R.params)
    rp = regionprops(e)[0]
    evs = rp.inertia_tensor_eigvals
    axis_lengths = _inertia_eigvals_to_axes_lengths_3D(evs)
    expected_lengths = sorted([2 * h for h in half_lengths], reverse=True)
    for ax_len_expected, ax_len in zip(expected_lengths, axis_lengths):
        assert abs(ax_len - ax_len_expected) < 0.01 * ax_len_expected
    assert abs(rp.axis_major_length - axis_lengths[0]) < 1e-07
    assert abs(rp.axis_minor_length - axis_lengths[-1]) < 1e-07