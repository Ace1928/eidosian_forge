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
@check_figures_equal(extensions=['png'])
@pytest.mark.parametrize('multi_value', ['BC', tuple('BC')])
def test_per_subplot_kw(self, fig_test, fig_ref, multi_value):
    x = 'AB;CD'
    grid_axes = fig_test.subplot_mosaic(x, subplot_kw={'facecolor': 'red'}, per_subplot_kw={'D': {'facecolor': 'blue'}, multi_value: {'facecolor': 'green'}})
    gs = fig_ref.add_gridspec(2, 2)
    for color, spec in zip(['red', 'green', 'green', 'blue'], gs):
        fig_ref.add_subplot(spec, facecolor=color)