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
def test_char_index_at():
    fig = plt.figure()
    text = fig.text(0.1, 0.9, '')
    text.set_text('i')
    bbox = text.get_window_extent()
    size_i = bbox.x1 - bbox.x0
    text.set_text('m')
    bbox = text.get_window_extent()
    size_m = bbox.x1 - bbox.x0
    text.set_text('iiiimmmm')
    bbox = text.get_window_extent()
    origin = bbox.x0
    assert text._char_index_at(origin - size_i) == 0
    assert text._char_index_at(origin) == 0
    assert text._char_index_at(origin + 0.499 * size_i) == 0
    assert text._char_index_at(origin + 0.501 * size_i) == 1
    assert text._char_index_at(origin + size_i * 3) == 3
    assert text._char_index_at(origin + size_i * 4 + size_m * 3) == 7
    assert text._char_index_at(origin + size_i * 4 + size_m * 4) == 8
    assert text._char_index_at(origin + size_i * 4 + size_m * 10) == 8