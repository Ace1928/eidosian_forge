from math import prod
from pathlib import Path
from unittest import skipUnless
import numpy as np
import pytest
from nibabel import pointset as ps
from nibabel.affines import apply_affine
from nibabel.arrayproxy import ArrayProxy
from nibabel.fileslice import strided_scalar
from nibabel.onetime import auto_attr
from nibabel.optpkg import optional_package
from nibabel.spatialimages import SpatialImage
from nibabel.tests.nibabel_data import get_nibabel_data
def test_GridIndices():
    shape = (2, 3)
    gi = ps.GridIndices(shape)
    assert gi.dtype == np.dtype('u1')
    assert gi.shape == (6, 2)
    assert repr(gi) == '<GridIndices(2, 3)>'
    gi_arr = np.asanyarray(gi)
    assert gi_arr.dtype == np.dtype('u1')
    assert gi_arr.shape == (6, 2)
    assert np.array_equal(gi_arr, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    shape = (2, 3, 4)
    gi = ps.GridIndices(shape)
    assert gi.dtype == np.dtype('u1')
    assert gi.shape == (24, 3)
    assert repr(gi) == '<GridIndices(2, 3, 4)>'
    gi_arr = np.asanyarray(gi)
    assert gi_arr.dtype == np.dtype('u1')
    assert gi_arr.shape == (24, 3)
    assert np.array_equal(gi_arr, np.mgrid[:2, :3, :4].reshape(3, -1).T)