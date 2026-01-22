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
def test_image_preserve_size():
    buff = io.BytesIO()
    im = np.zeros((481, 321))
    plt.imsave(buff, im, format='png')
    buff.seek(0)
    img = plt.imread(buff)
    assert img.shape[:2] == im.shape