import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
@check_figures_equal()
def test_marker_clipping(fig_ref, fig_test):
    marker_count = len(markers.MarkerStyle.markers)
    marker_size = 50
    ncol = 7
    nrow = marker_count // ncol + 1
    width = 2 * marker_size * ncol
    height = 2 * marker_size * nrow * 2
    fig_ref.set_size_inches((width / fig_ref.dpi, height / fig_ref.dpi))
    ax_ref = fig_ref.add_axes([0, 0, 1, 1])
    fig_test.set_size_inches((width / fig_test.dpi, height / fig_ref.dpi))
    ax_test = fig_test.add_axes([0, 0, 1, 1])
    for i, marker in enumerate(markers.MarkerStyle.markers):
        x = i % ncol
        y = i // ncol * 2
        ax_ref.plot([x, x], [y, y + 1], c='k', linestyle='-', lw=3)
        ax_ref.plot(x, y, c='k', marker=marker, markersize=marker_size, markeredgewidth=10, fillstyle='full', markerfacecolor='white')
        ax_ref.plot(x, y + 1, c='k', marker=marker, markersize=marker_size, markeredgewidth=10, fillstyle='full', markerfacecolor='white')
        ax_test.plot([x, x], [y, y + 1], c='k', linestyle='-', lw=3, marker=marker, markersize=marker_size, markeredgewidth=10, fillstyle='full', markerfacecolor='white')
    ax_ref.set(xlim=(-0.5, ncol), ylim=(-0.5, 2 * nrow))
    ax_test.set(xlim=(-0.5, ncol), ylim=(-0.5, 2 * nrow))
    ax_ref.axis('off')
    ax_test.axis('off')