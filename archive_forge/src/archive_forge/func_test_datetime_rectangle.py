import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
def test_datetime_rectangle():
    from datetime import datetime, timedelta
    start = datetime(2017, 1, 1, 0, 0, 0)
    delta = timedelta(seconds=16)
    patch = mpatches.Rectangle((start, 0), delta, 1)
    fig, ax = plt.subplots()
    ax.add_patch(patch)