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
@mpl.style.context('mpl20')
def test_picking_does_not_stale():
    fig, ax = plt.subplots()
    ax.scatter([0], [0], [1000], picker=True)
    fig.canvas.draw()
    assert not fig.stale
    mouse_event = SimpleNamespace(x=ax.bbox.x0 + ax.bbox.width / 2, y=ax.bbox.y0 + ax.bbox.height / 2, inaxes=ax, guiEvent=None)
    fig.pick(mouse_event)
    assert not fig.stale