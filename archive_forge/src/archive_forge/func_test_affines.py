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
@pytest.mark.parametrize('shape', [(5, 2), (5, 3), (5, 4)])
@pytest.mark.parametrize('homogeneous', [True, False])
def test_affines(self, shape, homogeneous):
    orig_coords = coords = self.rng.random(shape)
    if homogeneous:
        coords = np.column_stack([coords, np.ones(shape[0])])
    points = ps.Pointset(coords, homogeneous=homogeneous)
    assert np.allclose(points.get_coords(), orig_coords)
    scaler = np.diag([2] * shape[1] + [1])
    scaled = scaler @ points
    assert np.array_equal(scaled.coordinates, points.coordinates)
    assert np.array_equal(scaled.affine, scaler)
    assert np.allclose(scaled.get_coords(), 2 * orig_coords)
    flipper = np.eye(shape[1] + 1)
    flipper[:-1] = flipper[-2::-1]
    flipped = flipper @ points
    assert np.array_equal(flipped.coordinates, points.coordinates)
    assert np.array_equal(flipped.affine, flipper)
    assert np.allclose(flipped.get_coords(), orig_coords[:, ::-1])
    for doubledup in [scaler @ flipper @ points, scaler @ (flipper @ points)]:
        assert np.array_equal(doubledup.coordinates, points.coordinates)
        assert np.allclose(doubledup.affine, scaler @ flipper)
        assert np.allclose(doubledup.get_coords(), 2 * orig_coords[:, ::-1])