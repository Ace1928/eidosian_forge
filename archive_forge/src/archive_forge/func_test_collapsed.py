import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_collapsed():
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.annotate('BIG LONG STRING', xy=(1.25, 2), xytext=(10.5, 1.75), annotation_clip=False)
    p1 = ax.get_position()
    with pytest.warns(UserWarning):
        plt.tight_layout()
        p2 = ax.get_position()
        assert p1.width == p2.width
    with pytest.warns(UserWarning):
        plt.tight_layout(rect=[0, 0, 0.8, 0.8])