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
def test_invalid_arguments():
    t = mtransforms.Affine2D()
    with pytest.raises(ValueError):
        t.transform(1)
    with pytest.raises(ValueError):
        t.transform([[[1]]])
    with pytest.raises(RuntimeError):
        t.transform([])
    with pytest.raises(RuntimeError):
        t.transform([1])
    with pytest.raises(RuntimeError):
        t.transform([[1]])
    with pytest.raises(RuntimeError):
        t.transform([[1, 2, 3]])