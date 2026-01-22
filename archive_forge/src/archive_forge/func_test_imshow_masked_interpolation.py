from contextlib import ExitStack
from copy import copy
import functools
import io
import os
from pathlib import Path
import platform
import sys
import urllib.request
import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image
import matplotlib as mpl
from matplotlib import (
from matplotlib.image import (AxesImage, BboxImage, FigureImage,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.transforms import Bbox, Affine2D, TransformedBbox
import matplotlib.ticker as mticker
import pytest
@image_comparison(['imshow_masked_interpolation'], tol=0 if platform.machine() == 'x86_64' else 0.01, remove_text=True, style='mpl20')
def test_imshow_masked_interpolation():
    cmap = mpl.colormaps['viridis'].with_extremes(over='r', under='b', bad='k')
    N = 20
    n = colors.Normalize(vmin=0, vmax=N * N - 1)
    data = np.arange(N * N, dtype=float).reshape(N, N)
    data[5, 5] = -1
    data[15, 5] = 100000.0
    data[15, 15] = np.inf
    mask = np.zeros_like(data).astype('bool')
    mask[5, 15] = True
    data = np.ma.masked_array(data, mask)
    fig, ax_grid = plt.subplots(3, 6)
    interps = sorted(mimage._interpd_)
    interps.remove('antialiased')
    for interp, ax in zip(interps, ax_grid.ravel()):
        ax.set_title(interp)
        ax.imshow(data, norm=n, cmap=cmap, interpolation=interp)
        ax.axis('off')