from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION
def test_xkcd_no_cm():
    assert mpl.rcParams['path.sketch'] is None
    plt.xkcd()
    assert mpl.rcParams['path.sketch'] == (1, 100, 2)
    np.testing.break_cycles()
    assert mpl.rcParams['path.sketch'] == (1, 100, 2)