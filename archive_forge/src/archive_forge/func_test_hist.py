import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
def test_hist():
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(['a', 'b', 'a', 'c', 'ff'])
    assert n.shape == (10,)
    np.testing.assert_allclose(n, [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])