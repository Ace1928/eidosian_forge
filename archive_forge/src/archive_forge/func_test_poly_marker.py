import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
@check_figures_equal()
def test_poly_marker(fig_test, fig_ref):
    ax_test = fig_test.add_subplot()
    ax_ref = fig_ref.add_subplot()
    size = 20 ** 2
    ax_test.scatter([0], [0], marker=(4, 0, 45), s=size)
    ax_ref.scatter([0], [0], marker='s', s=size / 2)
    ax_test.scatter([1], [1], marker=(4, 0), s=size)
    ax_ref.scatter([1], [1], marker=UnsnappedMarkerStyle('D'), s=size / 2)
    ax_test.scatter([1], [1.5], marker=(4, 0, 0), s=size)
    ax_ref.scatter([1], [1.5], marker=UnsnappedMarkerStyle('D'), s=size / 2)
    ax_test.scatter([2], [2], marker=(5, 0), s=size)
    ax_ref.scatter([2], [2], marker=UnsnappedMarkerStyle('p'), s=size)
    ax_test.scatter([2], [2.5], marker=(5, 0, 0), s=size)
    ax_ref.scatter([2], [2.5], marker=UnsnappedMarkerStyle('p'), s=size)
    ax_test.scatter([3], [3], marker=(6, 0), s=size)
    ax_ref.scatter([3], [3], marker='h', s=size)
    ax_test.scatter([3], [3.5], marker=(6, 0, 0), s=size)
    ax_ref.scatter([3], [3.5], marker='h', s=size)
    ax_test.scatter([4], [4], marker=(6, 0, 30), s=size)
    ax_ref.scatter([4], [4], marker='H', s=size)
    ax_test.scatter([5], [5], marker=(8, 0, 22.5), s=size)
    ax_ref.scatter([5], [5], marker=UnsnappedMarkerStyle('8'), s=size)
    ax_test.set(xlim=(-0.5, 5.5), ylim=(-0.5, 5.5))
    ax_ref.set(xlim=(-0.5, 5.5), ylim=(-0.5, 5.5))