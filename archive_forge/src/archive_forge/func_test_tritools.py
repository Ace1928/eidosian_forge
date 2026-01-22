import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_tritools():
    x = np.array([0.0, 1.0, 0.5, 0.0, 2.0])
    y = np.array([0.0, 0.0, 0.5 * np.sqrt(3.0), -1.0, 1.0])
    triangles = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 4]], dtype=np.int32)
    mask = np.array([False, False, True], dtype=bool)
    triang = mtri.Triangulation(x, y, triangles, mask=mask)
    analyser = mtri.TriAnalyzer(triang)
    assert_array_almost_equal(analyser.scale_factors, [1, 1 / (1 + 3 ** 0.5 / 2)])
    assert_array_almost_equal(analyser.circle_ratios(rescale=False), np.ma.masked_array([0.5, 1.0 / (1.0 + np.sqrt(2.0)), np.nan], mask))
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 1.0 + 3.0, 1.0 + 6.0])
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    triang = mtri.Triangulation(x, y, triangles)
    analyser = mtri.TriAnalyzer(triang)
    assert_array_almost_equal(analyser.circle_ratios(), np.array([0.0]))
    n = 9

    def power(x, a):
        return np.abs(x) ** a * np.sign(x)
    x = np.linspace(-1.0, 1.0, n + 1)
    x, y = np.meshgrid(power(x, 2.0), power(x, 0.25))
    x = x.ravel()
    y = y.ravel()
    triang = mtri.Triangulation(x, y, triangles=meshgrid_triangles(n + 1))
    analyser = mtri.TriAnalyzer(triang)
    mask_flat = analyser.get_flat_tri_mask(0.2)
    verif_mask = np.zeros(162, dtype=bool)
    corners_index = [0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 34, 35, 126, 127, 142, 143, 144, 145, 146, 147, 158, 159, 160, 161]
    verif_mask[corners_index] = True
    assert_array_equal(mask_flat, verif_mask)
    mask = np.zeros(162, dtype=bool)
    mask[80] = True
    triang.set_mask(mask)
    mask_flat = analyser.get_flat_tri_mask(0.2)
    center_index = [44, 45, 62, 63, 78, 79, 80, 81, 82, 83, 98, 99, 116, 117]
    verif_mask[center_index] = True
    assert_array_equal(mask_flat, verif_mask)