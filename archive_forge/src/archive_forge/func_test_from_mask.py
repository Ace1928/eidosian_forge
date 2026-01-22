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
def test_from_mask(self):
    affine = np.diag([2, 3, 4, 1])
    mask = np.zeros((3, 3, 3))
    mask[1, 1, 1] = 1
    img = SpatialImage(mask, affine)
    grid = ps.Grid.from_mask(img)
    grid_coords = grid.get_coords()
    assert grid.n_coords == 1
    assert grid.dim == 3
    assert np.array_equal(grid_coords, [[2, 3, 4]])