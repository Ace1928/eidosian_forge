import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area
def test_masked_marching_cubes_empty():
    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.array([])
    with pytest.raises(ValueError):
        _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)