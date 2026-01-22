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
@pytest.mark.parametrize('x', [-1, 1])
@check_figures_equal(extensions=['png'])
def test_huge_range_log(fig_test, fig_ref, x):
    data = np.full((5, 5), x, dtype=np.float64)
    data[0:2, :] = 1e+20
    ax = fig_test.subplots()
    ax.imshow(data, norm=colors.LogNorm(vmin=1, vmax=data.max()), interpolation='nearest', cmap='viridis')
    data = np.full((5, 5), x, dtype=np.float64)
    data[0:2, :] = 1000
    ax = fig_ref.subplots()
    cmap = mpl.colormaps['viridis'].with_extremes(under='w')
    ax.imshow(data, norm=colors.Normalize(vmin=1, vmax=data.max()), interpolation='nearest', cmap=cmap)