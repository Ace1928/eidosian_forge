import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
def test_patch_linestyle_accents():
    star = mpath.Path.unit_regular_star(6)
    circle = mpath.Path.unit_circle()
    verts = np.concatenate([circle.vertices, star.vertices[::-1]])
    codes = np.concatenate([circle.codes, star.codes])
    linestyles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
    fig, ax = plt.subplots()
    for i, ls in enumerate(linestyles):
        star = mpath.Path(verts + i, codes)
        patch = mpatches.PathPatch(star, linewidth=3, linestyle=ls, facecolor=(1, 0, 0), edgecolor=(0, 0, 1))
        ax.add_patch(patch)
    ax.set_xlim([-1, i + 1])
    ax.set_ylim([-1, i + 1])
    fig.canvas.draw()