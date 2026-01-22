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
def test_patheffects():
    mpl.rcParams['path.effects'] = [patheffects.withStroke(linewidth=4, foreground='w')]
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    with io.BytesIO() as ps:
        fig.savefig(ps, format='ps')