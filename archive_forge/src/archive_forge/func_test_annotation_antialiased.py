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
def test_annotation_antialiased():
    annot = Annotation('foo\nbar', (0.5, 0.5), antialiased=True)
    assert annot._antialiased is True
    assert annot.get_antialiased() == annot._antialiased
    annot2 = Annotation('foo\nbar', (0.5, 0.5), antialiased=False)
    assert annot2._antialiased is False
    assert annot2.get_antialiased() == annot2._antialiased
    annot3 = Annotation('foo\nbar', (0.5, 0.5), antialiased=False)
    annot3.set_antialiased(True)
    assert annot3.get_antialiased() is True
    assert annot3._antialiased is True
    annot4 = Annotation('foo\nbar', (0.5, 0.5))
    assert annot4._antialiased == mpl.rcParams['text.antialiased']