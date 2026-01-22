import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
@image_comparison(['tight_layout1'], tol=1.9)
def test_tight_layout1():
    """Test tight_layout for a single subplot."""
    fig, ax = plt.subplots()
    example_plot(ax, fontsize=24)
    plt.tight_layout()