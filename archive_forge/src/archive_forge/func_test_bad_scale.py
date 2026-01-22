import copy
import matplotlib.pyplot as plt
from matplotlib.scale import (
import matplotlib.scale as mscale
from matplotlib.ticker import AsinhLocator, LogFormatterSciNotation
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import numpy as np
from numpy.testing import assert_allclose
import io
import pytest
def test_bad_scale(self):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        AsinhScale(axis=None, linear_width=0)
    with pytest.raises(ValueError):
        AsinhScale(axis=None, linear_width=-1)
    s0 = AsinhScale(axis=None)
    s1 = AsinhScale(axis=None, linear_width=3.0)