import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
def test_toolbar_zoompan():
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    ax = plt.gca()
    assert ax.get_navigate_mode() is None
    ax.figure.canvas.manager.toolmanager.trigger_tool('zoom')
    assert ax.get_navigate_mode() == 'ZOOM'
    ax.figure.canvas.manager.toolmanager.trigger_tool('pan')
    assert ax.get_navigate_mode() == 'PAN'