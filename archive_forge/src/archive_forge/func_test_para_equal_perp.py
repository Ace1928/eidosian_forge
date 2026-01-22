import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
@image_comparison(['para_equal_perp'], remove_text=True)
def test_para_equal_perp():
    x = np.array([0, 1, 2, 1, 0, -1, 0, 1] + [1] * 128)
    y = np.array([1, 1, 2, 1, 0, -1, 0, 0] + [0] * 128)
    fig, ax = plt.subplots()
    ax.plot(x + 1, y + 1)
    ax.plot(x + 1, y + 1, 'ro')