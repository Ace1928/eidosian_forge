import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
@check_figures_equal()
def test_closed_path_nan_removal(fig_test, fig_ref):
    ax_test = fig_test.subplots(2, 2).flatten()
    ax_ref = fig_ref.subplots(2, 2).flatten()
    path = Path([[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    ax_test[0].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, np.nan]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax_ref[0].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    ax_test[0].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax_ref[0].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, -3], [-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    ax_test[1].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, np.nan], [-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax_ref[1].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-1, -1], [1, -1], [1, np.nan], [0, 1], [-1, 1], [-1, -1]], [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY])
    ax_test[2].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-1, -1], [1, -1], [1, np.nan], [0, 1], [-1, 1], [-1, -1]], [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY])
    ax_ref[2].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, -3], [3, -3], [3, 0], [0, np.nan], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.LINETO])
    ax_test[2].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, -3], [3, -3], [3, 0], [0, np.nan], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.LINETO])
    ax_ref[2].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-1, -1], [1, -1], [1, np.nan], [0, 0], [0, 1], [-1, 1], [-1, -1]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY])
    ax_test[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-1, -1], [1, -1], [1, np.nan], [0, 0], [0, 1], [-1, 1], [-1, -1]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY])
    ax_ref[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-2, -2], [2, -2], [2, 0], [0, np.nan], [0, 2], [-2, 2], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.LINETO])
    ax_test[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-2, -2], [2, -2], [2, 0], [0, np.nan], [0, 2], [-2, 2], [-2, -2]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.LINETO])
    ax_ref[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, -3], [3, -3], [3, 0], [0, 0], [0, np.nan], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.LINETO])
    ax_test[3].add_patch(patches.PathPatch(path, facecolor='none'))
    path = Path([[-3, -3], [3, -3], [3, 0], [0, 0], [0, np.nan], [-3, 3], [-3, -3]], [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.LINETO])
    ax_ref[3].add_patch(patches.PathPatch(path, facecolor='none'))
    for ax in [*ax_test.flat, *ax_ref.flat]:
        ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
    remove_ticks_and_titles(fig_test)
    remove_ticks_and_titles(fig_ref)