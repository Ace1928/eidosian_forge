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
@check_figures_equal(extensions=['png'])
@pytest.mark.parametrize('str_pattern', ['AAA\nBBB', '\nAAA\nBBB\n', 'ABC\nDEF'])
def test_single_str_input(self, fig_test, fig_ref, str_pattern):
    grid_axes = fig_test.subplot_mosaic(str_pattern)
    grid_axes = fig_ref.subplot_mosaic([list(ln) for ln in str_pattern.strip().split('\n')])