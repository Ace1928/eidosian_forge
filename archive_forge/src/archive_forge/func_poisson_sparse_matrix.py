import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def poisson_sparse_matrix(n, m):
    """
        Return the sparse, (n*m, n*m) matrix in coo format resulting from the
        discretisation of the 2-dimensional Poisson equation according to a
        finite difference numerical scheme on a uniform (n, m) grid.
        """
    l = m * n
    rows = np.concatenate([np.arange(l, dtype=np.int32), np.arange(l - 1, dtype=np.int32), np.arange(1, l, dtype=np.int32), np.arange(l - n, dtype=np.int32), np.arange(n, l, dtype=np.int32)])
    cols = np.concatenate([np.arange(l, dtype=np.int32), np.arange(1, l, dtype=np.int32), np.arange(l - 1, dtype=np.int32), np.arange(n, l, dtype=np.int32), np.arange(l - n, dtype=np.int32)])
    vals = np.concatenate([4 * np.ones(l, dtype=np.float64), -np.ones(l - 1, dtype=np.float64), -np.ones(l - 1, dtype=np.float64), -np.ones(l - n, dtype=np.float64), -np.ones(l - n, dtype=np.float64)])
    vals[l:2 * l - 1][m - 1::m] = 0.0
    vals[2 * l - 1:3 * l - 2][m - 1::m] = 0.0
    return (vals, rows, cols, (n * m, n * m))