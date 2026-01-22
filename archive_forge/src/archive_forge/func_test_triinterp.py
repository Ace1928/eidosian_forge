import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_triinterp():
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    x = x.ravel()
    y = y.ravel()
    z = 1.23 * x - 4.79 * y
    triangles = [[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6], [3, 7, 6], [4, 5, 8], [5, 9, 8], [5, 6, 9], [6, 10, 9], [6, 7, 10], [7, 11, 10], [8, 9, 12], [9, 13, 12], [9, 10, 13], [10, 14, 13], [10, 11, 14], [11, 15, 14]]
    mask = np.zeros(len(triangles))
    mask[8:10] = 1
    triang = mtri.Triangulation(x, y, triangles, mask)
    linear_interp = mtri.LinearTriInterpolator(triang, z)
    cubic_min_E = mtri.CubicTriInterpolator(triang, z)
    cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
    xs = np.linspace(0.25, 2.75, 6)
    ys = [0.25, 0.75, 2.25, 2.75]
    xs, ys = np.meshgrid(xs, ys)
    for interp in (linear_interp, cubic_min_E, cubic_geom):
        zs = interp(xs, ys)
        assert_array_almost_equal(zs, 1.23 * xs - 4.79 * ys)
    xs = [-0.25, 1.25, 1.75, 3.25]
    ys = xs
    xs, ys = np.meshgrid(xs, ys)
    for interp in (linear_interp, cubic_min_E, cubic_geom):
        zs = linear_interp(xs, ys)
        assert_array_equal(zs.mask, [[True] * 4] * 4)
    xs = np.linspace(0.25, 1.75, 6)
    ys = [0.25, 0.75, 1.25, 1.75]
    xs, ys = np.meshgrid(xs, ys)
    for interp in (linear_interp, cubic_min_E, cubic_geom):
        zs = interp(xs, ys)
        matest.assert_array_almost_equal(zs, 1.23 * xs - 4.79 * ys)
        mask = (xs >= 1) * (xs <= 2) * (ys >= 1) * (ys <= 2)
        assert_array_equal(zs.mask, mask)
    a, b, c = (1.23, -4.79, 0.6)

    def quad(x, y):
        return a * (x - 0.5) ** 2 + b * (y - 0.5) ** 2 + c * x * y

    def gradient_quad(x, y):
        return (2 * a * (x - 0.5) + c * y, 2 * b * (y - 0.5) + c * x)
    x = np.array([0.2, 0.33367, 0.669, 0.0, 1.0, 1.0, 0.0])
    y = np.array([0.3, 0.80755, 0.4335, 0.0, 0.0, 1.0, 1.0])
    triangles = np.array([[0, 1, 2], [3, 0, 4], [4, 0, 2], [4, 2, 5], [1, 5, 2], [6, 5, 1], [6, 1, 0], [6, 0, 3]])
    triang = mtri.Triangulation(x, y, triangles)
    z = quad(x, y)
    dz = gradient_quad(x, y)
    xs = np.linspace(0.0, 1.0, 5)
    ys = np.linspace(0.0, 1.0, 5)
    xs, ys = np.meshgrid(xs, ys)
    cubic_user = mtri.CubicTriInterpolator(triang, z, kind='user', dz=dz)
    interp_zs = cubic_user(xs, ys)
    assert_array_almost_equal(interp_zs, quad(xs, ys))
    interp_dzsdx, interp_dzsdy = cubic_user.gradient(x, y)
    dzsdx, dzsdy = gradient_quad(x, y)
    assert_array_almost_equal(interp_dzsdx, dzsdx)
    assert_array_almost_equal(interp_dzsdy, dzsdy)
    n = 11
    x, y = np.meshgrid(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, n + 1))
    x = x.ravel()
    y = y.ravel()
    z = quad(x, y)
    triang = mtri.Triangulation(x, y, triangles=meshgrid_triangles(n + 1))
    xs, ys = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    xs = xs.ravel()
    ys = ys.ravel()
    linear_interp = mtri.LinearTriInterpolator(triang, z)
    cubic_min_E = mtri.CubicTriInterpolator(triang, z)
    cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
    zs = quad(xs, ys)
    diff_lin = np.abs(linear_interp(xs, ys) - zs)
    for interp in (cubic_min_E, cubic_geom):
        diff_cubic = np.abs(interp(xs, ys) - zs)
        assert np.max(diff_lin) >= 10 * np.max(diff_cubic)
        assert np.dot(diff_lin, diff_lin) >= 100 * np.dot(diff_cubic, diff_cubic)