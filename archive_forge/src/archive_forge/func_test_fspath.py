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
@pytest.mark.parametrize('fmt', ['png', 'pdf', 'ps', 'eps', 'svg'])
def test_fspath(fmt, tmpdir):
    out = Path(tmpdir, f'test.{fmt}')
    plt.savefig(out)
    with out.open('rb') as file:
        assert fmt.encode('ascii') in file.read(100).lower()