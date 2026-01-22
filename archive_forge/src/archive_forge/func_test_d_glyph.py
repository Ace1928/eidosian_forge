from collections import Counter
from pathlib import Path
import io
import re
import tempfile
import numpy as np
import pytest
from matplotlib import cbook, path, patheffects, font_manager as fm
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.testing._markers import needs_ghostscript, needs_usetex
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
@needs_ghostscript
def test_d_glyph(tmp_path):
    fig = plt.figure()
    fig.text(0.5, 0.5, 'def')
    out = tmp_path / 'test.eps'
    fig.savefig(out)
    mpl.testing.compare.convert(out, cache=False)