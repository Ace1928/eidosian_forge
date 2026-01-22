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
def test_affine_inverted_invalidated():
    point = [1.0, 1.0]
    t = mtransforms.Affine2D()
    assert_almost_equal(point, t.transform(t.inverted().transform(point)))
    t.translate(1.0, 1.0).get_matrix()
    assert_almost_equal(point, t.transform(t.inverted().transform(point)))