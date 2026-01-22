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
def test_bbox_intersection():
    bbox_from_ext = mtransforms.Bbox.from_extents
    inter = mtransforms.Bbox.intersection
    r1 = bbox_from_ext(0, 0, 1, 1)
    r2 = bbox_from_ext(0.5, 0.5, 1.5, 1.5)
    r3 = bbox_from_ext(0.5, 0, 0.75, 0.75)
    r4 = bbox_from_ext(0.5, 1.5, 1, 2.5)
    r5 = bbox_from_ext(1, 1, 2, 2)
    assert_bbox_eq(inter(r1, r1), r1)
    assert_bbox_eq(inter(r1, r2), bbox_from_ext(0.5, 0.5, 1, 1))
    assert_bbox_eq(inter(r1, r3), r3)
    assert inter(r1, r4) is None
    assert_bbox_eq(inter(r1, r5), bbox_from_ext(1, 1, 1, 1))