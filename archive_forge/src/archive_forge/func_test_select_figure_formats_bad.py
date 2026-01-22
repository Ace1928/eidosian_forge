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
def test_select_figure_formats_bad():
    ip = get_ipython()
    with pytest.raises(ValueError):
        pt.select_figure_formats(ip, 'foo')
    with pytest.raises(ValueError):
        pt.select_figure_formats(ip, {'png', 'foo'})
    with pytest.raises(ValueError):
        pt.select_figure_formats(ip, ['retina', 'pdf', 'bar', 'bad'])