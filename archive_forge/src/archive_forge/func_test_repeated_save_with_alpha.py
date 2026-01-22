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
def test_repeated_save_with_alpha():
    fig = Figure([1, 0.4])
    fig.set_facecolor((0, 1, 0.4))
    fig.patch.set_alpha(0.25)
    buf = io.BytesIO()
    fig.savefig(buf, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    fig.savefig(buf, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    assert_array_almost_equal(tuple(imread(buf)[0, 0]), (0.0, 1.0, 0.4, 0.25), decimal=3)