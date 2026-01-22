import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_internal_cpp_api():
    from matplotlib import _tri
    with pytest.raises(TypeError, match='__init__\\(\\): incompatible constructor arguments.'):
        mpl._tri.Triangulation()
    with pytest.raises(ValueError, match='x and y must be 1D arrays of the same length'):
        mpl._tri.Triangulation([], [1], [[]], (), (), (), False)
    x = [0, 1, 1]
    y = [0, 0, 1]
    with pytest.raises(ValueError, match='triangles must be a 2D array of shape \\(\\?,3\\)'):
        mpl._tri.Triangulation(x, y, [[0, 1]], (), (), (), False)
    tris = [[0, 1, 2]]
    with pytest.raises(ValueError, match='mask must be a 1D array with the same length as the triangles array'):
        mpl._tri.Triangulation(x, y, tris, [0, 1], (), (), False)
    with pytest.raises(ValueError, match='edges must be a 2D array with shape \\(\\?,2\\)'):
        mpl._tri.Triangulation(x, y, tris, (), [[1]], (), False)
    with pytest.raises(ValueError, match='neighbors must be a 2D array with the same shape as the triangles array'):
        mpl._tri.Triangulation(x, y, tris, (), (), [[-1]], False)
    triang = mpl._tri.Triangulation(x, y, tris, (), (), (), False)
    with pytest.raises(ValueError, match='z must be a 1D array with the same length as the triangulation x and y arrays'):
        triang.calculate_plane_coefficients([])
    for mask in ([0, 1], None):
        with pytest.raises(ValueError, match='mask must be a 1D array with the same length as the triangles array'):
            triang.set_mask(mask)
    triang.set_mask([True])
    assert_array_equal(triang.get_edges(), np.empty((0, 2)))
    triang.set_mask(())
    assert_array_equal(triang.get_edges(), [[1, 0], [2, 0], [2, 1]])
    with pytest.raises(TypeError, match='__init__\\(\\): incompatible constructor arguments.'):
        mpl._tri.TriContourGenerator()
    with pytest.raises(ValueError, match='z must be a 1D array with the same length as the x and y arrays'):
        mpl._tri.TriContourGenerator(triang, [1])
    z = [0, 1, 2]
    tcg = mpl._tri.TriContourGenerator(triang, z)
    with pytest.raises(ValueError, match='filled contour levels must be increasing'):
        tcg.create_filled_contour(1, 0)
    with pytest.raises(TypeError, match='__init__\\(\\): incompatible constructor arguments.'):
        mpl._tri.TrapezoidMapTriFinder()
    trifinder = mpl._tri.TrapezoidMapTriFinder(triang)
    with pytest.raises(ValueError, match='x and y must be array-like with same shape'):
        trifinder.find_many([0], [0, 1])