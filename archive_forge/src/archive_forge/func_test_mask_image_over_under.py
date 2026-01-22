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
@image_comparison(['mask_image_over_under.png'], remove_text=True, tol=1.0)
def test_mask_image_over_under():
    plt.rcParams['pcolormesh.snap'] = False
    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X ** 2 + Y ** 2) / 2) / (2 * np.pi)
    Z2 = np.exp(-(((X - 1) / 1.5) ** 2 + ((Y - 1) / 0.5) ** 2) / 2) / (2 * np.pi * 0.5 * 1.5)
    Z = 10 * (Z2 - Z1)
    palette = plt.cm.gray.with_extremes(over='r', under='g', bad='b')
    Zm = np.ma.masked_where(Z > 1.2, Z)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im = ax1.imshow(Zm, interpolation='bilinear', cmap=palette, norm=colors.Normalize(vmin=-1.0, vmax=1.0, clip=False), origin='lower', extent=[-3, 3, -3, 3])
    ax1.set_title('Green=low, Red=high, Blue=bad')
    fig.colorbar(im, extend='both', orientation='horizontal', ax=ax1, aspect=10)
    im = ax2.imshow(Zm, interpolation='nearest', cmap=palette, norm=colors.BoundaryNorm([-1, -0.5, -0.2, 0, 0.2, 0.5, 1], ncolors=256, clip=False), origin='lower', extent=[-3, 3, -3, 3])
    ax2.set_title('With BoundaryNorm')
    fig.colorbar(im, extend='both', spacing='proportional', orientation='horizontal', ax=ax2, aspect=10)