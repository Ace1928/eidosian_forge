import copy
from datetime import datetime
import io
from pathlib import Path
import pickle
import platform
from threading import Timer
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
from PIL import Image
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, FigureBase
from matplotlib.layout_engine import (ConstrainedLayoutEngine,
from matplotlib.ticker import AutoMinorLocator, FixedFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
@pytest.mark.parametrize('which', ['both', 'major', 'minor'])
def test_autofmt_xdate(which):
    date = ['3 Jan 2013', '4 Jan 2013', '5 Jan 2013', '6 Jan 2013', '7 Jan 2013', '8 Jan 2013', '9 Jan 2013', '10 Jan 2013', '11 Jan 2013', '12 Jan 2013', '13 Jan 2013', '14 Jan 2013']
    time = ['16:44:00', '16:45:00', '16:46:00', '16:47:00', '16:48:00', '16:49:00', '16:51:00', '16:52:00', '16:53:00', '16:55:00', '16:56:00', '16:57:00']
    angle = 60
    minors = [1, 2, 3, 4, 5, 6, 7]
    x = mdates.datestr2num(date)
    y = mdates.datestr2num(time)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.yaxis_date()
    ax.xaxis_date()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'FixedFormatter should only be used together with FixedLocator')
        ax.xaxis.set_minor_formatter(FixedFormatter(minors))
    fig.autofmt_xdate(0.2, angle, 'right', which)
    if which in ('both', 'major'):
        for label in fig.axes[0].get_xticklabels(False, 'major'):
            assert int(label.get_rotation()) == angle
    if which in ('both', 'minor'):
        for label in fig.axes[0].get_xticklabels(True, 'minor'):
            assert int(label.get_rotation()) == angle