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
@image_comparison(['image_alpha'], remove_text=True)
def test_image_alpha():
    np.random.seed(0)
    Z = np.random.rand(6, 6)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(Z, alpha=1.0, interpolation='none')
    ax2.imshow(Z, alpha=0.5, interpolation='none')
    ax3.imshow(Z, alpha=0.5, interpolation='nearest')