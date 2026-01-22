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
def test_figureoptions_with_datetime_axes():
    fig, ax = plt.subplots()
    xydata = [datetime(year=2021, month=1, day=1), datetime(year=2021, month=2, day=1)]
    ax.plot(xydata, xydata)
    with mock.patch('matplotlib.backends.qt_compat._exec', lambda obj: None):
        fig.canvas.manager.toolbar.edit_parameters()