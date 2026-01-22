import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_bbox_frozen_copies_minpos():
    bbox = mtransforms.Bbox.from_extents(0.0, 0.0, 1.0, 1.0, minpos=1.0)
    frozen = bbox.frozen()
    assert_array_equal(frozen.minpos, bbox.minpos)