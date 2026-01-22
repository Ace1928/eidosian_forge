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
def test_align_labels_stray_axes():
    fig, axs = plt.subplots(2, 2)
    for nn, ax in enumerate(axs.flat):
        ax.set_xlabel('Boo')
        ax.set_xlabel('Who')
        ax.plot(np.arange(4) ** nn, np.arange(4) ** nn)
    fig.align_ylabels()
    fig.align_xlabels()
    fig.draw_without_rendering()
    xn = np.zeros(4)
    yn = np.zeros(4)
    for nn, ax in enumerate(axs.flat):
        yn[nn] = ax.xaxis.label.get_position()[1]
        xn[nn] = ax.yaxis.label.get_position()[0]
    np.testing.assert_allclose(xn[:2], xn[2:])
    np.testing.assert_allclose(yn[::2], yn[1::2])
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for nn, ax in enumerate(axs.flat):
        ax.set_xlabel('Boo')
        ax.set_xlabel('Who')
        pc = ax.pcolormesh(np.random.randn(10, 10))
    fig.colorbar(pc, ax=ax)
    fig.align_ylabels()
    fig.align_xlabels()
    fig.draw_without_rendering()
    xn = np.zeros(4)
    yn = np.zeros(4)
    for nn, ax in enumerate(axs.flat):
        yn[nn] = ax.xaxis.label.get_position()[1]
        xn[nn] = ax.yaxis.label.get_position()[0]
    np.testing.assert_allclose(xn[:2], xn[2:])
    np.testing.assert_allclose(yn[::2], yn[1::2])