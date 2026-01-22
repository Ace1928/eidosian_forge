import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
@image_comparison(['tight_layout9'])
def test_tight_layout9():
    f, axarr = plt.subplots(2, 2)
    axarr[1][1].set_visible(False)
    plt.tight_layout()