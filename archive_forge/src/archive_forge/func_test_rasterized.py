import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree
import xml.parsers.expat
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib import font_manager as fm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
@check_figures_equal(tol=20)
def test_rasterized(fig_test, fig_ref):
    t = np.arange(0, 100) * 2.3
    x = np.cos(t)
    y = np.sin(t)
    ax_ref = fig_ref.subplots()
    ax_ref.plot(x, y, '-', c='r', lw=10)
    ax_ref.plot(x + 1, y, '-', c='b', lw=10)
    ax_test = fig_test.subplots()
    ax_test.plot(x, y, '-', c='r', lw=10, rasterized=True)
    ax_test.plot(x + 1, y, '-', c='b', lw=10, rasterized=True)