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
def test_iterability_axes_argument():

    class MyAxes(Axes):

        def __init__(self, *args, myclass=None, **kwargs):
            Axes.__init__(self, *args, **kwargs)

    class MyClass:

        def __getitem__(self, item):
            if item != 'a':
                raise ValueError('item should be a')

        def _as_mpl_axes(self):
            return (MyAxes, {'myclass': self})
    fig = plt.figure()
    fig.add_subplot(1, 1, 1, projection=MyClass())
    plt.close(fig)