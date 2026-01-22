from datetime import datetime
import io
import warnings
import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib.text import Text, Annotation, OffsetFrom
def test_text_stale():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.draw_all()
    assert not ax1.stale
    assert not ax2.stale
    assert not fig.stale
    txt1 = ax1.text(0.5, 0.5, 'aardvark')
    assert ax1.stale
    assert txt1.stale
    assert fig.stale
    ann1 = ax2.annotate('aardvark', xy=[0.5, 0.5])
    assert ax2.stale
    assert ann1.stale
    assert fig.stale
    plt.draw_all()
    assert not ax1.stale
    assert not ax2.stale
    assert not fig.stale