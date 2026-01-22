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
def test_non_tuple_rgbaface():
    fig = plt.figure()
    fig.add_subplot(projection='3d').scatter([0, 1, 2], [0, 1, 2], path_effects=[patheffects.Stroke(linewidth=4)])
    fig.canvas.draw()