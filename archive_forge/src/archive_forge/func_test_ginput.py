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
def test_ginput(recwarn):
    warnings.filterwarnings('ignore', 'cannot show the figure')
    fig, ax = plt.subplots()
    trans = ax.transData.transform

    def single_press():
        MouseEvent('button_press_event', fig.canvas, *trans((0.1, 0.2)), 1)._process()
    Timer(0.1, single_press).start()
    assert fig.ginput() == [(0.1, 0.2)]

    def multi_presses():
        MouseEvent('button_press_event', fig.canvas, *trans((0.1, 0.2)), 1)._process()
        KeyEvent('key_press_event', fig.canvas, 'backspace')._process()
        MouseEvent('button_press_event', fig.canvas, *trans((0.3, 0.4)), 1)._process()
        MouseEvent('button_press_event', fig.canvas, *trans((0.5, 0.6)), 1)._process()
        MouseEvent('button_press_event', fig.canvas, *trans((0, 0)), 2)._process()
    Timer(0.1, multi_presses).start()
    np.testing.assert_allclose(fig.ginput(3), [(0.3, 0.4), (0.5, 0.6)])