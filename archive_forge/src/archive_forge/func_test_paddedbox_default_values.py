from collections import namedtuple
import io
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.offsetbox import (
def test_paddedbox_default_values():
    fig, ax = plt.subplots()
    at = AnchoredText('foo', 'upper left')
    pb = PaddedBox(at, patch_attrs={'facecolor': 'r'}, draw_frame=True)
    ax.add_artist(pb)
    fig.draw_without_rendering()