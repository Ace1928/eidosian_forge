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
def test_suptitle_subfigures():
    fig = plt.figure(figsize=(4, 3))
    sf1, sf2 = fig.subfigures(1, 2)
    sf2.set_facecolor('white')
    sf1.subplots()
    sf2.subplots()
    fig.suptitle('This is a visible suptitle.')
    assert sf1.get_facecolor() == (0.0, 0.0, 0.0, 0.0)
    assert sf2.get_facecolor() == (1.0, 1.0, 1.0, 1.0)