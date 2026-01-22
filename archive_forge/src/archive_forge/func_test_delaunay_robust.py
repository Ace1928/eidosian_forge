import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_delaunay_robust():
    tri_points = np.array([[0.8660254037844384, -0.5000000000000004], [0.7577722283113836, -0.5000000000000004], [0.6495190528383288, -0.5000000000000003], [0.5412658773652739, -0.5000000000000003], [0.811898816047911, -0.40625000000000044], [0.7036456405748561, -0.4062500000000004], [0.5953924651018013, -0.40625000000000033]])
    test_points = np.asarray([[0.58, -0.46], [0.65, -0.46], [0.65, -0.42], [0.7, -0.48], [0.7, -0.44], [0.75, -0.44], [0.8, -0.48]])

    def tri_contains_point(xtri, ytri, xy):
        tri_points = np.vstack((xtri, ytri)).T
        return Path(tri_points).contains_point(xy)

    def tris_contain_point(triang, xy):
        return sum((tri_contains_point(triang.x[tri], triang.y[tri], xy) for tri in triang.triangles))
    triang = mtri.Triangulation(tri_points[:, 0], tri_points[:, 1])
    for test_point in test_points:
        assert tris_contain_point(triang, test_point) == 1
    triang = mtri.Triangulation(tri_points[1:, 0], tri_points[1:, 1])