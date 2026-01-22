import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_shared_and_moved():
    f, (a1, a2) = plt.subplots(1, 2, sharey=True)
    check_ticklabel_visible([a2], [True], [False])
    a2.yaxis.tick_left()
    check_ticklabel_visible([a2], [True], [False])
    f, (a1, a2) = plt.subplots(2, 1, sharex=True)
    check_ticklabel_visible([a1], [False], [True])
    a2.xaxis.tick_bottom()
    check_ticklabel_visible([a1], [False], [True])