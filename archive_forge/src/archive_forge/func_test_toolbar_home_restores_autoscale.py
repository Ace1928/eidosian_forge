import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
def test_toolbar_home_restores_autoscale():
    fig, ax = plt.subplots()
    ax.plot(range(11), range(11))
    tb = NavigationToolbar2(fig.canvas)
    tb.zoom()
    KeyEvent('key_press_event', fig.canvas, 'k', 100, 100)._process()
    KeyEvent('key_press_event', fig.canvas, 'l', 100, 100)._process()
    assert ax.get_xlim() == ax.get_ylim() == (1, 10)
    KeyEvent('key_press_event', fig.canvas, 'k', 100, 100)._process()
    KeyEvent('key_press_event', fig.canvas, 'l', 100, 100)._process()
    assert ax.get_xlim() == ax.get_ylim() == (0, 10)
    start, stop = ax.transData.transform([(2, 2), (5, 5)])
    MouseEvent('button_press_event', fig.canvas, *start, MouseButton.LEFT)._process()
    MouseEvent('button_release_event', fig.canvas, *stop, MouseButton.LEFT)._process()
    KeyEvent('key_press_event', fig.canvas, 'h')._process()
    assert ax.get_xlim() == ax.get_ylim() == (0, 10)
    KeyEvent('key_press_event', fig.canvas, 'k', 100, 100)._process()
    KeyEvent('key_press_event', fig.canvas, 'l', 100, 100)._process()
    assert ax.get_xlim() == ax.get_ylim() == (1, 10)