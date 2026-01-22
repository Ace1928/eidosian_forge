import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_trianalyzer_mismatched_indices():
    x = np.array([0.0, 1.0, 0.5, 0.0, 2.0])
    y = np.array([0.0, 0.0, 0.5 * np.sqrt(3.0), -1.0, 1.0])
    triangles = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 4]], dtype=np.int32)
    mask = np.array([False, False, True], dtype=bool)
    triang = mtri.Triangulation(x, y, triangles, mask=mask)
    analyser = mtri.TriAnalyzer(triang)
    analyser._get_compressed_triangulation()