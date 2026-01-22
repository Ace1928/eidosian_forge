import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_trirefiner_fortran_contiguous_triangles():
    triangles1 = np.array([[2, 0, 3], [2, 1, 0]])
    assert not np.isfortran(triangles1)
    triangles2 = np.array(triangles1, copy=True, order='F')
    assert np.isfortran(triangles2)
    x = np.array([0.39, 0.59, 0.43, 0.32])
    y = np.array([33.99, 34.01, 34.19, 34.18])
    triang1 = mtri.Triangulation(x, y, triangles1)
    triang2 = mtri.Triangulation(x, y, triangles2)
    refiner1 = mtri.UniformTriRefiner(triang1)
    refiner2 = mtri.UniformTriRefiner(triang2)
    fine_triang1 = refiner1.refine_triangulation(subdiv=1)
    fine_triang2 = refiner2.refine_triangulation(subdiv=1)
    assert_array_equal(fine_triang1.triangles, fine_triang2.triangles)