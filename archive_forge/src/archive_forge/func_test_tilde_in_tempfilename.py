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
@needs_usetex
@needs_ghostscript
def test_tilde_in_tempfilename(tmpdir):
    base_tempdir = Path(tmpdir, 'short-1')
    base_tempdir.mkdir()
    with cbook._setattr_cm(tempfile, tempdir=str(base_tempdir)):
        mpl.rcParams['text.usetex'] = True
        plt.plot([1, 2, 3, 4])
        plt.xlabel('\\textbf{time} (s)')
        plt.savefig(base_tempdir / 'tex_demo.eps', format='ps')