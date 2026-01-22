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
def test_str_norms(fig_test, fig_ref):
    t = np.random.rand(10, 10) * 0.8 + 0.1
    axts = fig_test.subplots(1, 5)
    axts[0].imshow(t, norm='log')
    axts[1].imshow(t, norm='log', vmin=0.2)
    axts[2].imshow(t, norm='symlog')
    axts[3].imshow(t, norm='symlog', vmin=0.3, vmax=0.7)
    axts[4].imshow(t, norm='logit', vmin=0.3, vmax=0.7)
    axrs = fig_ref.subplots(1, 5)
    axrs[0].imshow(t, norm=colors.LogNorm())
    axrs[1].imshow(t, norm=colors.LogNorm(vmin=0.2))
    axrs[2].imshow(t, norm=colors.SymLogNorm(linthresh=2))
    axrs[3].imshow(t, norm=colors.SymLogNorm(linthresh=2, vmin=0.3, vmax=0.7))
    axrs[4].imshow(t, norm='logit', clim=(0.3, 0.7))
    assert type(axts[0].images[0].norm) is colors.LogNorm
    with pytest.raises(ValueError):
        axts[0].imshow(t, norm='foobar')