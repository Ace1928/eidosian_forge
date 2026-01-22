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
@image_comparison(['interp_alpha.png'], remove_text=True)
def test_alpha_interp():
    """Test the interpolation of the alpha channel on RGBA images"""
    fig, (axl, axr) = plt.subplots(1, 2)
    img = np.zeros((5, 5, 4))
    img[..., 1] = np.ones((5, 5))
    img[..., 3] = np.tril(np.ones((5, 5), dtype=np.uint8))
    axl.imshow(img, interpolation='none')
    axr.imshow(img, interpolation='bilinear')