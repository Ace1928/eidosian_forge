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
def test_transformedbbox_contains():
    bb = TransformedBbox(Bbox.unit(), Affine2D().rotate_deg(30))
    assert bb.contains(0.8, 0.5)
    assert bb.contains(-0.4, 0.85)
    assert not bb.contains(0.9, 0.5)
    bb = TransformedBbox(Bbox.unit(), Affine2D().translate(0.25, 0.5))
    assert bb.contains(1.25, 1.5)
    assert not bb.fully_contains(1.25, 1.5)
    assert not bb.fully_contains(0.1, 0.1)