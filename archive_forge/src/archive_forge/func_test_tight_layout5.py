import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
@image_comparison(['tight_layout5'])
def test_tight_layout5():
    """Test tight_layout for image."""
    ax = plt.subplot()
    arr = np.arange(100).reshape((10, 10))
    ax.imshow(arr, interpolation='none')
    plt.tight_layout()