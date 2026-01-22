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
def test_offset_copy_errors():
    with pytest.raises(ValueError, match="'fontsize' is not a valid value for units; supported values are 'dots', 'points', 'inches'"):
        mtransforms.offset_copy(None, units='fontsize')
    with pytest.raises(ValueError, match='For units of inches or points a fig kwarg is needed'):
        mtransforms.offset_copy(None, units='inches')