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
@check_figures_equal()
def test_imshow_pil(fig_test, fig_ref):
    style.use('default')
    png_path = Path(__file__).parent / 'baseline_images/pngsuite/basn3p04.png'
    tiff_path = Path(__file__).parent / 'baseline_images/test_image/uint16.tif'
    axs = fig_test.subplots(2)
    axs[0].imshow(Image.open(png_path))
    axs[1].imshow(Image.open(tiff_path))
    axs = fig_ref.subplots(2)
    axs[0].imshow(plt.imread(png_path))
    axs[1].imshow(plt.imread(tiff_path))