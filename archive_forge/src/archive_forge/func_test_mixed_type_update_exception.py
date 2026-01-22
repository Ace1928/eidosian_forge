import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('plotter', plotters)
@pytest.mark.parametrize('xdata', fvalues, ids=fids)
def test_mixed_type_update_exception(self, plotter, xdata):
    ax = plt.figure().subplots()
    with pytest.raises(TypeError):
        plotter(ax, [0, 3], [1, 3])
        plotter(ax, xdata, [1, 2])