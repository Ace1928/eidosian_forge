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
def test_image_cursor_formatting():
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((4, 4)))
    data = np.ma.masked_array([0], mask=[True])
    assert im.format_cursor_data(data) == '[]'
    data = np.ma.masked_array([0], mask=[False])
    assert im.format_cursor_data(data) == '[0]'
    data = np.nan
    assert im.format_cursor_data(data) == '[nan]'