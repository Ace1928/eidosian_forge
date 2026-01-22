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
def test_add_subplot_kwargs():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax1 = fig.add_subplot(1, 1, 1)
    assert ax is not None
    assert ax1 is not ax
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    ax1 = fig.add_subplot(1, 1, 1, projection='polar')
    assert ax is not None
    assert ax1 is not ax
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    ax1 = fig.add_subplot(1, 1, 1)
    assert ax is not None
    assert ax1.name == 'rectilinear'
    assert ax1 is not ax
    plt.close()