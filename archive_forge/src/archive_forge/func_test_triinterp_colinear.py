import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_triinterp_colinear():
    delta = 0.0
    x0 = np.array([1.5, 0, 1, 2, 3, 1.5, 1.5])
    y0 = np.array([-1, 0, 0, 0, 0, delta, 1])
    transformations = [[1, 0], [0, 1], [1, 1], [1, 2], [-2, -1], [-2, 1]]
    for transformation in transformations:
        x_rot = transformation[0] * x0 + transformation[1] * y0
        y_rot = -transformation[1] * x0 + transformation[0] * y0
        x, y = (x_rot, y_rot)
        z = 1.23 * x - 4.79 * y
        triangles = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [1, 2, 5], [2, 3, 5], [3, 4, 5], [1, 5, 6], [4, 6, 5]]
        triang = mtri.Triangulation(x, y, triangles)
        xs = np.linspace(np.min(triang.x), np.max(triang.x), 20)
        ys = np.linspace(np.min(triang.y), np.max(triang.y), 20)
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.ravel()
        ys = ys.ravel()
        mask_out = triang.get_trifinder()(xs, ys) == -1
        zs_target = np.ma.array(1.23 * xs - 4.79 * ys, mask=mask_out)
        linear_interp = mtri.LinearTriInterpolator(triang, z)
        cubic_min_E = mtri.CubicTriInterpolator(triang, z)
        cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
        for interp in (linear_interp, cubic_min_E, cubic_geom):
            zs = interp(xs, ys)
            assert_array_almost_equal(zs_target, zs)
        itri = 4
        pt1 = triang.triangles[itri, 0]
        pt2 = triang.triangles[itri, 1]
        xs = np.linspace(triang.x[pt1], triang.x[pt2], 10)
        ys = np.linspace(triang.y[pt1], triang.y[pt2], 10)
        zs_target = 1.23 * xs - 4.79 * ys
        for interp in (linear_interp, cubic_min_E, cubic_geom):
            zs, = interp._interpolate_multikeys(xs, ys, tri_index=itri * np.ones(10, dtype=np.int32))
            assert_array_almost_equal(zs_target, zs)