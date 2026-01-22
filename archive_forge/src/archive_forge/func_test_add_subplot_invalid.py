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
def test_add_subplot_invalid():
    fig = plt.figure()
    with pytest.raises(ValueError, match='Number of columns must be a positive integer'):
        fig.add_subplot(2, 0, 1)
    with pytest.raises(ValueError, match='Number of rows must be a positive integer'):
        fig.add_subplot(0, 2, 1)
    with pytest.raises(ValueError, match='num must be an integer with 1 <= num <= 4'):
        fig.add_subplot(2, 2, 0)
    with pytest.raises(ValueError, match='num must be an integer with 1 <= num <= 4'):
        fig.add_subplot(2, 2, 5)
    with pytest.raises(ValueError, match='num must be an integer with 1 <= num <= 4'):
        fig.add_subplot(2, 2, 0.5)
    with pytest.raises(ValueError, match='must be a three-digit integer'):
        fig.add_subplot(42)
    with pytest.raises(ValueError, match='must be a three-digit integer'):
        fig.add_subplot(1000)
    with pytest.raises(TypeError, match='takes 1 or 3 positional arguments but 2 were given'):
        fig.add_subplot(2, 2)
    with pytest.raises(TypeError, match='takes 1 or 3 positional arguments but 4 were given'):
        fig.add_subplot(1, 2, 3, 4)
    with pytest.raises(ValueError, match="Number of rows must be a positive integer, not '2'"):
        fig.add_subplot('2', 2, 1)
    with pytest.raises(ValueError, match='Number of columns must be a positive integer, not 2.0'):
        fig.add_subplot(2, 2.0, 1)
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match='The Axes must have been created in the present figure'):
        fig.add_subplot(ax)