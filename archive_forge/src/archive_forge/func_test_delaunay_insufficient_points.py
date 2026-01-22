import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@pytest.mark.parametrize('x, y', [([], []), ([1], [5]), ([1, 2], [5, 6]), ([1, 2, 1], [5, 6, 5]), ([1, 2, 2], [5, 6, 6]), ([1, 1, 1, 2, 1, 2], [5, 5, 5, 6, 5, 6])])
def test_delaunay_insufficient_points(x, y):
    with pytest.raises(ValueError):
        mtri.Triangulation(x, y)