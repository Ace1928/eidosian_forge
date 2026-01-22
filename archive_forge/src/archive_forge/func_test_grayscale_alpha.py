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
@image_comparison(['grayscale_alpha.pdf'])
def test_grayscale_alpha():
    """Masking images with NaN did not work for grayscale images"""
    x, y = np.ogrid[-2:2:0.1, -2:2:0.1]
    dd = np.exp(-(x ** 2 + y ** 2))
    dd[dd < 0.1] = np.nan
    fig, ax = plt.subplots()
    ax.imshow(dd, interpolation='none', cmap='gray_r')
    ax.set_xticks([])
    ax.set_yticks([])