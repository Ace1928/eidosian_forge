import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
def test_toolmanager_remove():
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.gcf()
    initial_len = len(fig.canvas.manager.toolmanager.tools)
    assert 'forward' in fig.canvas.manager.toolmanager.tools
    fig.canvas.manager.toolmanager.remove_tool('forward')
    assert len(fig.canvas.manager.toolmanager.tools) == initial_len - 1
    assert 'forward' not in fig.canvas.manager.toolmanager.tools