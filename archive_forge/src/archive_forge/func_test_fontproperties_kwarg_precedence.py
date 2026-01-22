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
def test_fontproperties_kwarg_precedence():
    """Test that kwargs take precedence over fontproperties defaults."""
    plt.figure()
    text1 = plt.xlabel('value', fontproperties='Times New Roman', size=40.0)
    text2 = plt.ylabel('counts', size=40.0, fontproperties='Times New Roman')
    assert text1.get_size() == 40.0
    assert text2.get_size() == 40.0