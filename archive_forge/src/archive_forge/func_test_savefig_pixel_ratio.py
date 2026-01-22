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
@pytest.mark.parametrize('backend', [pytest.param('Agg', marks=[pytest.mark.backend('Agg')]), pytest.param('Cairo', marks=[pytest.mark.backend('Cairo')])])
def test_savefig_pixel_ratio(backend):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        ratio1 = Image.open(buf)
        ratio1.load()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    fig.canvas._set_device_pixel_ratio(2)
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        ratio2 = Image.open(buf)
        ratio2.load()
    assert ratio1 == ratio2