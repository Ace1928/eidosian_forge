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
def test_image_preserve_size2():
    n = 7
    data = np.identity(n, float)
    fig = plt.figure(figsize=(n, n), frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, interpolation='nearest', origin='lower', aspect='auto')
    buff = io.BytesIO()
    fig.savefig(buff, dpi=1)
    buff.seek(0)
    img = plt.imread(buff)
    assert img.shape == (7, 7, 4)
    assert_array_equal(np.asarray(img[:, :, 0], bool), np.identity(n, bool)[::-1])