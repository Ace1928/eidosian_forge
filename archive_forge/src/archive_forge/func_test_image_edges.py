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
def test_image_edges():
    fig = plt.figure(figsize=[1, 1])
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    data = np.tile(np.arange(12), 15).reshape(20, 9)
    im = ax.imshow(data, origin='upper', extent=[-10, 10, -10, 10], interpolation='none', cmap='gray')
    x = y = 2
    ax.set_xlim([-x, x])
    ax.set_ylim([-y, y])
    ax.set_xticks([])
    ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, facecolor=(0, 1, 0))
    buf.seek(0)
    im = plt.imread(buf)
    r, g, b, a = sum(im[:, 0])
    r, g, b, a = sum(im[:, -1])
    assert g != 100, 'Expected a non-green edge - but sadly, it was.'