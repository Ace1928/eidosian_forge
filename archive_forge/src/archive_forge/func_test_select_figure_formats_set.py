from binascii import a2b_base64
from io import BytesIO
import pytest
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import numpy as np
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import _PNG, _JPEG
from .. import pylabtools as pt
from IPython.testing import decorators as dec
def test_select_figure_formats_set():
    ip = get_ipython()
    for fmts in [{'png', 'svg'}, ['png'], ('jpeg', 'pdf', 'retina'), {'svg'}]:
        active_mimes = {_fmt_mime_map[fmt] for fmt in fmts}
        pt.select_figure_formats(ip, fmts)
        for mime, f in ip.display_formatter.formatters.items():
            if mime in active_mimes:
                assert Figure in f
            else:
                assert Figure not in f