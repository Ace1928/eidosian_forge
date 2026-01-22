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
@mpl.style.context('default')
@check_figures_equal(extensions=['pdf', 'eps'])
def test_pdf_eps_savefig_when_color_is_none(fig_test, fig_ref):
    ax_test = fig_test.add_subplot()
    ax_test.set_axis_off()
    ax_test.plot(np.sin(np.linspace(-5, 5, 100)), 'v', c='none')
    ax_ref = fig_ref.add_subplot()
    ax_ref.set_axis_off()