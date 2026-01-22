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
def test_axesimage_get_shape():
    ax = plt.gca()
    im = AxesImage(ax)
    with pytest.raises(RuntimeError, match='You must first set the image array'):
        im.get_shape()
    z = np.arange(12, dtype=float).reshape((4, 3))
    im.set_data(z)
    assert im.get_shape() == (4, 3)
    assert im.get_size() == im.get_shape()