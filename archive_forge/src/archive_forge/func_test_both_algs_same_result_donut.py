import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area
def test_both_algs_same_result_donut():
    n = 48
    a, b = (2.5 / n, -1.25)
    vol = np.empty((n, n, n), 'float32')
    for iz in range(vol.shape[0]):
        for iy in range(vol.shape[1]):
            for ix in range(vol.shape[2]):
                z, y, x = (float(iz) * a + b, float(iy) * a + b, float(ix) * a + b)
                vol[iz, iy, ix] = (((8 * x) ** 2 + (8 * y - 2) ** 2 + (8 * z) ** 2 + 16 - 1.85 * 1.85) * ((8 * x) ** 2 + (8 * y - 2) ** 2 + (8 * z) ** 2 + 16 - 1.85 * 1.85) - 64 * ((8 * x) ** 2 + (8 * y - 2) ** 2)) * (((8 * x) ** 2 + (8 * y - 2 + 4) * (8 * y - 2 + 4) + (8 * z) ** 2 + 16 - 1.85 * 1.85) * ((8 * x) ** 2 + (8 * y - 2 + 4) * (8 * y - 2 + 4) + (8 * z) ** 2 + 16 - 1.85 * 1.85) - 64 * ((8 * y - 2 + 4) * (8 * y - 2 + 4) + (8 * z) ** 2)) + 1025
    vertices1, faces1 = marching_cubes(vol, 0, method='lorensen')[:2]
    vertices2, faces2 = marching_cubes(vol, 0)[:2]
    assert not _same_mesh(vertices1, faces1, vertices2, faces2)