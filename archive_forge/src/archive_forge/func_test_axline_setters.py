import itertools
import platform
import timeit
from types import SimpleNamespace
from cycler import cycler
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import _path
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_axline_setters():
    fig, ax = plt.subplots()
    line1 = ax.axline((0.1, 0.1), slope=0.6)
    line2 = ax.axline((0.1, 0.1), (0.8, 0.4))
    line1.set_xy1(0.2, 0.3)
    line1.set_slope(2.4)
    line2.set_xy1(0.3, 0.2)
    line2.set_xy2(0.6, 0.8)
    assert line1.get_xy1() == (0.2, 0.3)
    assert line1.get_slope() == 2.4
    assert line2.get_xy1() == (0.3, 0.2)
    assert line2.get_xy2() == (0.6, 0.8)
    with pytest.raises(ValueError, match="Cannot set an 'xy2' value while 'slope' is set"):
        line1.set_xy2(0.2, 0.3)
    with pytest.raises(ValueError, match="Cannot set a 'slope' value while 'xy2' is set"):
        line2.set_slope(3)