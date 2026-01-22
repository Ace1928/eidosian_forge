import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area
def test_both_algs_same_result_ellipse():
    sphere_small = ellipsoid(1, 1, 1, levelset=True)
    vertices1, faces1 = marching_cubes(sphere_small, 0, allow_degenerate=False)[:2]
    vertices2, faces2 = marching_cubes(sphere_small, 0, allow_degenerate=False, method='lorensen')[:2]
    assert _same_mesh(vertices1, faces1, vertices2, faces2)