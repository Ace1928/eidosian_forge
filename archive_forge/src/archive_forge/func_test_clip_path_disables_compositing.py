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
@check_figures_equal(extensions=['pdf'])
def test_clip_path_disables_compositing(fig_test, fig_ref):
    t = np.arange(9).reshape((3, 3))
    for fig in [fig_test, fig_ref]:
        ax = fig.add_subplot()
        ax.imshow(t, clip_path=(mpl.path.Path([(0, 0), (0, 1), (1, 0)]), ax.transData))
        ax.imshow(t, clip_path=(mpl.path.Path([(1, 1), (1, 2), (2, 1)]), ax.transData))
    fig_ref.suppressComposite = True