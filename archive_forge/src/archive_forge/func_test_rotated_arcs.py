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
@image_comparison(['all_quadrants_arcs.svg'], style='mpl20')
def test_rotated_arcs():
    fig, ax_arr = plt.subplots(2, 2, squeeze=False, figsize=(10, 10))
    scale = 10000000
    diag_centers = ((-1, -1), (-1, 1), (1, 1), (1, -1))
    on_axis_centers = ((0, 1), (1, 0), (0, -1), (-1, 0))
    skews = ((2, 2), (2, 1 / 10), (2, 1 / 100), (2, 1 / 1000))
    for ax, (sx, sy) in zip(ax_arr.ravel(), skews):
        k = 0
        for prescale, centers in zip((1 - 0.0001, (1 - 0.0001) / np.sqrt(2)), (on_axis_centers, diag_centers)):
            for j, (x_sign, y_sign) in enumerate(centers, start=k):
                a = Arc((x_sign * scale * prescale, y_sign * scale * prescale), scale * sx, scale * sy, lw=4, color=f'C{j}', zorder=1 + j, angle=np.rad2deg(np.arctan2(y_sign, x_sign)) % 360, label=f'big {j}', gid=f'big {j}')
                ax.add_patch(a)
            k = j + 1
        ax.set_xlim(-scale / 4000, scale / 4000)
        ax.set_ylim(-scale / 4000, scale / 4000)
        ax.axhline(0, color='k')
        ax.axvline(0, color='k')
        ax.set_axis_off()
        ax.set_aspect('equal')