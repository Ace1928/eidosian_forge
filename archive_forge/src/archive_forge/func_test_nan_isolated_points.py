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
@image_comparison(['nan_path'], style='default', remove_text=True, extensions=['pdf', 'svg', 'eps', 'png'])
def test_nan_isolated_points():
    y0 = [0, np.nan, 2, np.nan, 4, 5, 6]
    y1 = [np.nan, 7, np.nan, 9, 10, np.nan, 12]
    fig, ax = plt.subplots()
    ax.plot(y0, '-o')
    ax.plot(y1, '-o')