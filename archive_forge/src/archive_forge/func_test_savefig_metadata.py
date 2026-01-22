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
def test_savefig_metadata(monkeypatch):
    pikepdf = pytest.importorskip('pikepdf')
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '0')
    fig, ax = plt.subplots()
    ax.plot(range(5))
    md = {'Author': 'me', 'Title': 'Multipage PDF', 'Subject': 'Test page', 'Keywords': 'test,pdf,multipage', 'ModDate': datetime.datetime(1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))), 'Trapped': 'True'}
    buf = io.BytesIO()
    fig.savefig(buf, metadata=md, format='pdf')
    with pikepdf.Pdf.open(buf) as pdf:
        info = {k: str(v) for k, v in pdf.docinfo.items()}
    assert info == {'/Author': 'me', '/CreationDate': 'D:19700101000000Z', '/Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org', '/Keywords': 'test,pdf,multipage', '/ModDate': 'D:19680801000000Z', '/Producer': f'Matplotlib pdf backend v{mpl.__version__}', '/Subject': 'Test page', '/Title': 'Multipage PDF', '/Trapped': '/True'}