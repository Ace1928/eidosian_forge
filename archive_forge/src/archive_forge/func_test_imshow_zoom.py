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
@check_figures_equal(extensions=['png'])
def test_imshow_zoom(fig_test, fig_ref):
    np.random.seed(19680801)
    dpi = plt.rcParams['savefig.dpi']
    A = np.random.rand(int(dpi * 3), int(dpi * 3))
    for fig in [fig_test, fig_ref]:
        fig.set_size_inches(2.9, 2.9)
    ax = fig_test.subplots()
    ax.imshow(A, interpolation='antialiased')
    ax.set_xlim([10, 20])
    ax.set_ylim([10, 20])
    ax = fig_ref.subplots()
    ax.imshow(A, interpolation='nearest')
    ax.set_xlim([10, 20])
    ax.set_ylim([10, 20])