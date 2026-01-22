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
def test_bbox_as_strings():
    b = mtransforms.Bbox([[0.5, 0], [0.75, 0.75]])
    assert_bbox_eq(b, eval(repr(b), {'Bbox': mtransforms.Bbox}))
    asdict = eval(str(b), {'Bbox': dict})
    for k, v in asdict.items():
        assert getattr(b, k) == v
    fmt = '.1f'
    asdict = eval(format(b, fmt), {'Bbox': dict})
    for k, v in asdict.items():
        assert eval(format(getattr(b, k), fmt)) == v