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
@pytest.mark.parametrize('img_size, fig_size, interpolation', [(5, 2, 'hanning'), (5, 5, 'nearest'), (5, 10, 'nearest'), (3, 2.9, 'hanning'), (3, 9.1, 'nearest')])
@check_figures_equal(extensions=['png'])
def test_imshow_antialiased(fig_test, fig_ref, img_size, fig_size, interpolation):
    np.random.seed(19680801)
    dpi = plt.rcParams['savefig.dpi']
    A = np.random.rand(int(dpi * img_size), int(dpi * img_size))
    for fig in [fig_test, fig_ref]:
        fig.set_size_inches(fig_size, fig_size)
    ax = fig_test.subplots()
    ax.set_position([0, 0, 1, 1])
    ax.imshow(A, interpolation='antialiased')
    ax = fig_ref.subplots()
    ax.set_position([0, 0, 1, 1])
    ax.imshow(A, interpolation=interpolation)