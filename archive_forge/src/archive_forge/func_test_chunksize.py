import io
import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image, TiffTags
import pytest
from matplotlib import (
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison
from matplotlib.transforms import IdentityTransform
def test_chunksize():
    x = range(200)
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x))
    fig.canvas.draw()
    fig, ax = plt.subplots()
    rcParams['agg.path.chunksize'] = 105
    ax.plot(x, np.sin(x))
    fig.canvas.draw()