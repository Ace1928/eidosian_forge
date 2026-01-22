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
@dec.skip_without('PIL.Image')
def test_figure_to_jpeg():
    _check_pil_jpeg_bytes()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3])
    plt.draw()
    jpeg = pt.print_figure(fig, 'jpeg', pil_kwargs={'optimize': 50})[:100].lower()
    assert jpeg.startswith(_JPEG)