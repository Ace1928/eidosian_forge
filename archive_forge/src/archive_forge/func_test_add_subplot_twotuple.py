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
def test_add_subplot_twotuple():
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, (3, 5))
    assert ax1.get_subplotspec().rowspan == range(1, 3)
    assert ax1.get_subplotspec().colspan == range(0, 1)
    ax2 = fig.add_subplot(3, 2, (4, 6))
    assert ax2.get_subplotspec().rowspan == range(1, 3)
    assert ax2.get_subplotspec().colspan == range(1, 2)
    ax3 = fig.add_subplot(3, 2, (3, 6))
    assert ax3.get_subplotspec().rowspan == range(1, 3)
    assert ax3.get_subplotspec().colspan == range(0, 2)
    ax4 = fig.add_subplot(3, 2, (4, 5))
    assert ax4.get_subplotspec().rowspan == range(1, 3)
    assert ax4.get_subplotspec().colspan == range(0, 2)
    with pytest.raises(IndexError):
        fig.add_subplot(3, 2, (6, 3))