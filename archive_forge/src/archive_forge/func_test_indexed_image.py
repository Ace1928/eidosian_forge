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
def test_indexed_image():
    pikepdf = pytest.importorskip('pikepdf')
    data = np.zeros((256, 1, 3), dtype=np.uint8)
    data[:, 0, 0] = np.arange(256)
    rcParams['pdf.compression'] = True
    fig = plt.figure()
    fig.figimage(data, resize=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', dpi='figure')
    with pikepdf.Pdf.open(buf) as pdf:
        page, = pdf.pages
        image, = page.images.values()
        pdf_image = pikepdf.PdfImage(image)
        assert pdf_image.indexed
        pil_image = pdf_image.as_pil_image()
        rgb = np.asarray(pil_image.convert('RGB'))
    np.testing.assert_array_equal(data, rgb)