import copy
import importlib
import os
import signal
import sys
from datetime import date, datetime
from unittest import mock
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib import _c_internal_utils
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_double_resize():
    fig, ax = plt.subplots()
    fig.canvas.draw()
    window = fig.canvas.manager.window
    w, h = (3, 2)
    fig.set_size_inches(w, h)
    assert fig.canvas.width() == w * matplotlib.rcParams['figure.dpi']
    assert fig.canvas.height() == h * matplotlib.rcParams['figure.dpi']
    old_width = window.width()
    old_height = window.height()
    fig.set_size_inches(w, h)
    assert window.width() == old_width
    assert window.height() == old_height