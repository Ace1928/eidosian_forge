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
def test_text_urls():
    pikepdf = pytest.importorskip('pikepdf')
    test_url = 'https://test_text_urls.matplotlib.org/'
    fig = plt.figure(figsize=(2, 1))
    fig.text(0.1, 0.1, 'test plain 123', url=f'{test_url}plain')
    fig.text(0.1, 0.4, 'test mathtext $123$', url=f'{test_url}mathtext')
    with io.BytesIO() as fd:
        fig.savefig(fd, format='pdf')
        with pikepdf.Pdf.open(fd) as pdf:
            annots = pdf.pages[0].Annots
            for y, fragment in [('0.1', 'plain'), ('0.4', 'mathtext')]:
                annot = next((a for a in annots if a.A.URI == f'{test_url}{fragment}'), None)
                assert annot is not None
                assert getattr(annot, 'QuadPoints', None) is None
                assert annot.Rect[1] == decimal.Decimal(y) * 72