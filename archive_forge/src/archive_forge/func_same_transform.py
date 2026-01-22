import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..affines import from_matvec, to_matvec
from ..orientations import (
from ..testing import deprecated_to, expires
def same_transform(taff, ornt, shape):
    shape = np.array(shape)
    size = np.prod(shape)
    arr = np.arange(size).reshape(shape)
    t_arr = apply_orientation(arr, ornt)
    i, j, k = shape
    arr_pts = np.mgrid[:i, :j, :k].reshape((3, -1))
    itaff = np.linalg.inv(taff)
    o2t_pts = np.dot(itaff[:3, :3], arr_pts) + itaff[:3, 3][:, None]
    assert np.allclose(np.round(o2t_pts), o2t_pts)
    vals = t_arr[tuple(o2t_pts.astype('i'))]
    return np.all(vals == arr.ravel())