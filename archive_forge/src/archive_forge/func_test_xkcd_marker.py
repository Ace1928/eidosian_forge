import re
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.backend_bases import MouseEvent
@image_comparison(['xkcd_marker.png'], remove_text=True)
def test_xkcd_marker():
    np.random.seed(0)
    x = np.linspace(0, 5, 8)
    y1 = x
    y2 = 5 - x
    y3 = 2.5 * np.ones(8)
    with plt.xkcd():
        fig, ax = plt.subplots()
        ax.plot(x, y1, '+', ms=10)
        ax.plot(x, y2, 'o', ms=10)
        ax.plot(x, y3, '^', ms=10)