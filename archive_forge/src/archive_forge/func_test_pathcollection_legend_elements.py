from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_pathcollection_legend_elements():
    np.random.seed(19680801)
    x, y = np.random.rand(2, 10)
    y = np.random.rand(10)
    c = np.random.randint(0, 5, size=10)
    s = np.random.randint(10, 300, size=10)
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=c, s=s, cmap='jet', marker='o', linewidths=0)
    h, l = sc.legend_elements(fmt='{x:g}')
    assert len(h) == 5
    assert l == ['0', '1', '2', '3', '4']
    colors = np.array([line.get_color() for line in h])
    colors2 = sc.cmap(np.arange(5) / 4)
    assert_array_equal(colors, colors2)
    l1 = ax.legend(h, l, loc=1)
    h2, lab2 = sc.legend_elements(num=9)
    assert len(h2) == 9
    l2 = ax.legend(h2, lab2, loc=2)
    h, l = sc.legend_elements(prop='sizes', alpha=0.5, color='red')
    assert all((line.get_alpha() == 0.5 for line in h))
    assert all((line.get_markerfacecolor() == 'red' for line in h))
    l3 = ax.legend(h, l, loc=4)
    h, l = sc.legend_elements(prop='sizes', num=4, fmt='{x:.2f}', func=lambda x: 2 * x)
    actsizes = [line.get_markersize() for line in h]
    labeledsizes = np.sqrt(np.array(l, float) / 2)
    assert_array_almost_equal(actsizes, labeledsizes)
    l4 = ax.legend(h, l, loc=3)
    loc = mpl.ticker.MaxNLocator(nbins=9, min_n_ticks=9 - 1, steps=[1, 2, 2.5, 3, 5, 6, 8, 10])
    h5, lab5 = sc.legend_elements(num=loc)
    assert len(h2) == len(h5)
    levels = [-1, 0, 55.4, 260]
    h6, lab6 = sc.legend_elements(num=levels, prop='sizes', fmt='{x:g}')
    assert [float(l) for l in lab6] == levels[2:]
    for l in [l1, l2, l3, l4]:
        ax.add_artist(l)
    fig.canvas.draw()