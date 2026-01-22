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
@pytest.mark.parametrize('clear_meth', ['clear', 'clf'])
def test_figure_clear(clear_meth):
    fig = plt.figure()
    fig.clear()
    assert fig.axes == []
    ax = fig.add_subplot(111)
    getattr(fig, clear_meth)()
    assert fig.axes == []
    axes = [fig.add_subplot(2, 1, i + 1) for i in range(2)]
    getattr(fig, clear_meth)()
    assert fig.axes == []
    gs = fig.add_gridspec(ncols=2, nrows=1)
    subfig = fig.add_subfigure(gs[0])
    subaxes = subfig.add_subplot(111)
    getattr(fig, clear_meth)()
    assert subfig not in fig.subfigs
    assert fig.axes == []
    subfig = fig.add_subfigure(gs[0])
    subaxes = subfig.add_subplot(111)
    mainaxes = fig.add_subplot(gs[1])
    mainaxes.remove()
    assert fig.axes == [subaxes]
    mainaxes = fig.add_subplot(gs[1])
    subaxes.remove()
    assert fig.axes == [mainaxes]
    assert subfig in fig.subfigs
    subaxes = subfig.add_subplot(111)
    assert mainaxes in fig.axes
    assert subaxes in fig.axes
    getattr(subfig, clear_meth)()
    assert subfig in fig.subfigs
    assert subaxes not in subfig.axes
    assert subaxes not in fig.axes
    assert mainaxes in fig.axes
    subaxes = subfig.add_subplot(111)
    getattr(fig, clear_meth)()
    assert fig.axes == []
    assert fig.subfigs == []
    subfigs = [fig.add_subfigure(gs[i]) for i in [0, 1]]
    subaxes = [sfig.add_subplot(111) for sfig in subfigs]
    assert all((ax in fig.axes for ax in subaxes))
    assert all((sfig in fig.subfigs for sfig in subfigs))
    getattr(subfigs[0], clear_meth)()
    assert subaxes[0] not in fig.axes
    assert subaxes[1] in fig.axes
    assert subfigs[1] in fig.subfigs
    getattr(subfigs[1], clear_meth)()
    subfigs = [fig.add_subfigure(gs[i]) for i in [0, 1]]
    subaxes = [sfig.add_subplot(111) for sfig in subfigs]
    assert all((ax in fig.axes for ax in subaxes))
    assert all((sfig in fig.subfigs for sfig in subfigs))
    getattr(fig, clear_meth)()
    assert fig.subfigs == []
    assert fig.axes == []