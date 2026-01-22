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
@image_comparison(['pdf_use14corefonts.pdf'])
def test_use14corefonts():
    rcParams['pdf.use14corefonts'] = True
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 8
    rcParams['font.sans-serif'] = ['Helvetica']
    rcParams['pdf.compression'] = 0
    text = 'A three-line text positioned just above a blue line\nand containing some French characters and the euro symbol:\n"Merci pépé pour les 10 €"'
    fig, ax = plt.subplots()
    ax.set_title('Test PDF backend with option use14corefonts=True')
    ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='bottom', fontsize=14)
    ax.axhline(0.5, linewidth=0.5)