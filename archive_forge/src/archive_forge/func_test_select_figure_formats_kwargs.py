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
def test_select_figure_formats_kwargs():
    ip = get_ipython()
    kwargs = dict(bbox_inches='tight')
    pt.select_figure_formats(ip, 'png', **kwargs)
    formatter = ip.display_formatter.formatters['image/png']
    f = formatter.lookup_by_type(Figure)
    cell = f.keywords
    expected = kwargs
    expected['base64'] = True
    expected['fmt'] = 'png'
    assert cell == expected
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3])
    plt.draw()
    formatter.enabled = True
    png = formatter(fig)
    assert isinstance(png, str)
    png_bytes = a2b_base64(png)
    assert png_bytes.startswith(_PNG)