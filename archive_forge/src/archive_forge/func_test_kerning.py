import datetime
import decimal
import io
import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import (
from matplotlib.cbook import _get_data_path
from matplotlib.ft2font import FT2Font
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.backends._backend_pdf_ps import get_glyphs_subset
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
@image_comparison(['kerning.pdf'])
def test_kerning():
    fig = plt.figure()
    s = 'AVAVAVAVAVAVAVAV€AAVV'
    fig.text(0, 0.25, s, size=5)
    fig.text(0, 0.75, s, size=20)