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
@mpl.rc_context({'savefig.transparent': True})
@check_figures_equal(extensions=['png'])
def test_savefig_transparent(fig_test, fig_ref):
    gs1 = fig_test.add_gridspec(3, 3, left=0.05, wspace=0.05)
    f1 = fig_test.add_subfigure(gs1[:, :])
    f2 = f1.add_subfigure(gs1[0, 0])
    ax12 = f2.add_subplot(gs1[:, :])
    ax1 = f1.add_subplot(gs1[:-1, :])
    iax1 = ax1.inset_axes([0.1, 0.2, 0.3, 0.4])
    iax2 = iax1.inset_axes([0.1, 0.2, 0.3, 0.4])
    ax2 = fig_test.add_subplot(gs1[-1, :-1])
    ax3 = fig_test.add_subplot(gs1[-1, -1])
    for ax in [ax12, ax1, iax1, iax2, ax2, ax3]:
        ax.set(xticks=[], yticks=[])
        ax.spines[:].set_visible(False)