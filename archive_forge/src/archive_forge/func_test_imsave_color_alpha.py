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
def test_imsave_color_alpha():
    np.random.seed(1)
    for origin in ['lower', 'upper']:
        data = np.random.rand(16, 16, 4)
        buff = io.BytesIO()
        plt.imsave(buff, data, origin=origin, format='png')
        buff.seek(0)
        arr_buf = plt.imread(buff)
        data = (255 * data).astype('uint8')
        if origin == 'lower':
            data = data[::-1]
        arr_buf = (255 * arr_buf).astype('uint8')
        assert_array_equal(data, arr_buf)