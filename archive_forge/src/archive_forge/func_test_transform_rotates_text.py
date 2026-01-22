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
def test_transform_rotates_text():
    ax = plt.gca()
    transform = mtransforms.Affine2D().rotate_deg(30)
    text = ax.text(0, 0, 'test', transform=transform, transform_rotates_text=True)
    result = text.get_rotation()
    assert_almost_equal(result, 30)