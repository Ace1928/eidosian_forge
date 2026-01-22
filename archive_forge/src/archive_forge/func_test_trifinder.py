import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_trifinder():
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    x = x.ravel()
    y = y.ravel()
    triangles = [[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6], [3, 7, 6], [4, 5, 8], [5, 9, 8], [5, 6, 9], [6, 10, 9], [6, 7, 10], [7, 11, 10], [8, 9, 12], [9, 13, 12], [9, 10, 13], [10, 14, 13], [10, 11, 14], [11, 15, 14]]
    mask = np.zeros(len(triangles))
    mask[8:10] = 1
    triang = mtri.Triangulation(x, y, triangles, mask)
    trifinder = triang.get_trifinder()
    xs = [0.25, 1.25, 2.25, 3.25]
    ys = [0.25, 1.25, 2.25, 3.25]
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.ravel()
    ys = ys.ravel()
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [0, 2, 4, -1, 6, -1, 10, -1, 12, 14, 16, -1, -1, -1, -1, -1])
    tris = trifinder(xs - 0.5, ys - 0.5)
    assert_array_equal(tris, [-1, -1, -1, -1, -1, 1, 3, 5, -1, 7, -1, 11, -1, 13, 15, 17])
    xs = [0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 1.5, 1.5, 0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.0, 2.0, 1.5, 1.5, 1.5, 1.5]
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [0, 2, 4, 13, 15, 17, 3, 14, 6, 7, 10, 11])
    xs = [0.0, 3.0]
    ys = [0.0, 3.0]
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [0, 17])
    delta = 0.0
    x = [1.5, 0, 1, 2, 3, 1.5, 1.5]
    y = [-1, 0, 0, 0, 0, delta, 1]
    triangles = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [1, 2, 5], [2, 3, 5], [3, 4, 5], [1, 5, 6], [4, 6, 5]]
    triang = mtri.Triangulation(x, y, triangles)
    trifinder = triang.get_trifinder()
    xs = [-0.1, 0.4, 0.9, 1.4, 1.9, 2.4, 2.9]
    ys = [-0.1, 0.1]
    xs, ys = np.meshgrid(xs, ys)
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [[-1, 0, 0, 1, 1, 2, -1], [-1, 6, 6, 6, 7, 7, -1]])
    delta = 0.0
    x = [-1, -delta, 0, 0, 0, 0, 1]
    y = [1.5, 1.5, 0, 1, 2, 3, 1.5]
    triangles = [[0, 1, 2], [0, 1, 5], [1, 2, 3], [1, 3, 4], [1, 4, 5], [2, 6, 3], [3, 6, 4], [4, 6, 5]]
    triang = mtri.Triangulation(x, y, triangles)
    trifinder = triang.get_trifinder()
    xs = [-0.1, 0.1]
    ys = [-0.1, 0.4, 0.9, 1.4, 1.9, 2.4, 2.9]
    xs, ys = np.meshgrid(xs, ys)
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [[-1, -1], [0, 5], [0, 5], [0, 6], [1, 6], [1, 7], [-1, -1]])
    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    triangles = [[0, 1, 2], [1, 3, 2]]
    triang = mtri.Triangulation(x, y, triangles)
    trifinder = triang.get_trifinder()
    xs = [-0.2, 0.2, 0.8, 1.2]
    ys = [0.5, 0.5, 0.5, 0.5]
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [-1, 0, 1, -1])
    triang.set_mask([1, 0])
    assert trifinder == triang.get_trifinder()
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [-1, -1, 1, -1])