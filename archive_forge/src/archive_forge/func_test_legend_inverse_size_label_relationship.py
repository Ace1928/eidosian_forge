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
def test_legend_inverse_size_label_relationship():
    """
    Ensure legend markers scale appropriately when label and size are
    inversely related.
    Here label = 5 / size
    """
    np.random.seed(19680801)
    X = np.random.random(50)
    Y = np.random.random(50)
    C = 1 - np.random.random(50)
    S = 5 / C
    legend_sizes = [0.2, 0.4, 0.6, 0.8]
    fig, ax = plt.subplots()
    sc = ax.scatter(X, Y, s=S)
    handles, labels = sc.legend_elements(prop='sizes', num=legend_sizes, func=lambda s: 5 / s)
    handle_sizes = [x.get_markersize() for x in handles]
    handle_sizes = [5 / x ** 2 for x in handle_sizes]
    assert_array_almost_equal(handle_sizes, legend_sizes, decimal=1)