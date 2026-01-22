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
def test_unclipped():
    fig, ax = plt.subplots()
    ax.set_axis_off()
    im = ax.imshow([[0, 0], [0, 0]], aspect='auto', extent=(-10, 10, -10, 10), cmap='gray', clip_on=False)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.canvas.draw()
    assert (np.array(fig.canvas.buffer_rgba())[..., :3] == 0).all()