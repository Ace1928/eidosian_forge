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
def test_gca():
    fig = plt.figure()
    ax0 = fig.add_axes([0, 0, 1, 1])
    assert fig.gca() is ax0
    ax1 = fig.add_subplot(111)
    assert fig.gca() is ax1
    fig.add_axes(ax0)
    assert fig.axes == [ax0, ax1]
    assert fig.gca() is ax0
    fig.sca(ax0)
    assert fig.axes == [ax0, ax1]
    fig.add_subplot(ax1)
    assert fig.axes == [ax0, ax1]
    assert fig.gca() is ax1