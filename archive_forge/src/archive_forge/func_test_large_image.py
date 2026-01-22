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
@pytest.mark.filterwarnings('ignore:Data with more than .* cannot be accurately displayed')
@pytest.mark.parametrize('origin', ['upper', 'lower'])
@pytest.mark.parametrize('dim, size, msg', [['row', 2 ** 23, '2\\*\\*23 columns'], ['col', 2 ** 24, '2\\*\\*24 rows']])
@check_figures_equal(extensions=('png',))
def test_large_image(fig_test, fig_ref, dim, size, msg, origin):
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()
    array = np.zeros((1, size + 2))
    array[:, array.size // 2:] = 1
    if dim == 'col':
        array = array.T
    im = ax_test.imshow(array, vmin=0, vmax=1, aspect='auto', extent=(0, 1, 0, 1), interpolation='none', origin=origin)
    with pytest.warns(UserWarning, match=f'Data with more than {msg} cannot be accurately displayed.'):
        fig_test.canvas.draw()
    array = np.zeros((1, 2))
    array[:, 1] = 1
    if dim == 'col':
        array = array.T
    im = ax_ref.imshow(array, vmin=0, vmax=1, aspect='auto', extent=(0, 1, 0, 1), interpolation='none', origin=origin)