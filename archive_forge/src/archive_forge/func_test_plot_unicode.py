import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('plotter', PLOT_LIST, ids=PLOT_IDS)
def test_plot_unicode(self, plotter):
    ax = plt.figure().subplots()
    words = ['Здравствуйте', 'привет']
    plotter(ax, words, [0, 1])
    axis_test(ax.xaxis, words)