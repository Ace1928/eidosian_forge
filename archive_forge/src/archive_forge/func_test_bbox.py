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
def test_bbox():
    fig, ax = plt.subplots()
    with io.BytesIO() as buf:
        fig.savefig(buf, format='eps')
        buf = buf.getvalue()
    bb = re.search(b'^%%BoundingBox: (.+) (.+) (.+) (.+)$', buf, re.MULTILINE)
    assert bb
    hibb = re.search(b'^%%HiResBoundingBox: (.+) (.+) (.+) (.+)$', buf, re.MULTILINE)
    assert hibb
    for i in range(1, 5):
        assert b'.' not in bb.group(i)
        assert int(bb.group(i)) == pytest.approx(float(hibb.group(i)), 1)