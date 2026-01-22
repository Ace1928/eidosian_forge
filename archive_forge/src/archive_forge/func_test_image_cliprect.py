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
@image_comparison(['image_cliprect'], style='mpl20')
def test_image_cliprect():
    fig, ax = plt.subplots()
    d = [[1, 2], [3, 4]]
    im = ax.imshow(d, extent=(0, 5, 0, 5))
    rect = patches.Rectangle(xy=(1, 1), width=2, height=2, transform=im.axes.transData)
    im.set_clip_path(rect)