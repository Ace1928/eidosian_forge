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
@image_comparison(['test_subfigure_ss.png'], style='mpl20', savefig_kwarg={'facecolor': 'teal'}, tol=0.02)
def test_subfigure_ss():
    np.random.seed(19680801)
    fig = plt.figure(layout='constrained')
    gs = fig.add_gridspec(1, 2)
    sub = fig.add_subfigure(gs[0], facecolor='pink')
    axs = sub.subplots(2, 2)
    for ax in axs.flat:
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2, vmax=2)
    sub.colorbar(pc, ax=axs)
    sub.suptitle('Left Side')
    ax = fig.add_subplot(gs[1])
    ax.plot(np.arange(20))
    ax.set_title('Axes')
    fig.suptitle('Figure suptitle', fontsize='xx-large')