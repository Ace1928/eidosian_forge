import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_delaunay():
    nx = 5
    ny = 4
    x, y = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
    x = x.ravel()
    y = y.ravel()
    npoints = nx * ny
    ntriangles = 2 * (nx - 1) * (ny - 1)
    nedges = 3 * nx * ny - 2 * nx - 2 * ny + 1
    triang = mtri.Triangulation(x, y)
    assert_array_almost_equal(triang.x, x)
    assert_array_almost_equal(triang.y, y)
    assert len(triang.triangles) == ntriangles
    assert np.min(triang.triangles) == 0
    assert np.max(triang.triangles) == npoints - 1
    assert len(triang.edges) == nedges
    assert np.min(triang.edges) == 0
    assert np.max(triang.edges) == npoints - 1
    neighbors = triang.neighbors
    triang._neighbors = None
    assert_array_equal(triang.neighbors, neighbors)
    assert_array_equal(np.unique(triang.triangles), np.arange(npoints))