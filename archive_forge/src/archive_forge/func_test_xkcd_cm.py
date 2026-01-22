from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION
def test_xkcd_cm():
    assert mpl.rcParams['path.sketch'] is None
    with plt.xkcd():
        assert mpl.rcParams['path.sketch'] == (1, 100, 2)
    assert mpl.rcParams['path.sketch'] is None