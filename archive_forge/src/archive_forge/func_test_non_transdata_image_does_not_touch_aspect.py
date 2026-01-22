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
def test_non_transdata_image_does_not_touch_aspect():
    ax = plt.figure().add_subplot()
    im = np.arange(4).reshape((2, 2))
    ax.imshow(im, transform=ax.transAxes)
    assert ax.get_aspect() == 'auto'
    ax.imshow(im, transform=Affine2D().scale(2) + ax.transData)
    assert ax.get_aspect() == 1
    ax.imshow(im, transform=ax.transAxes, aspect=2)
    assert ax.get_aspect() == 2