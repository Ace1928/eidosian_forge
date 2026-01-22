import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@pytest.mark.parametrize('remove_ticks', [True, False])
def test_label_outer(remove_ticks):
    f, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for ax in axs.flat:
        ax.set(xlabel='foo', ylabel='bar')
        ax.label_outer(remove_inner_ticks=remove_ticks)
    check_ticklabel_visible(axs.flat, [False, False, True, True], [True, False, True, False])
    if remove_ticks:
        check_tick1_visible(axs.flat, [False, False, True, True], [True, False, True, False])
    else:
        check_tick1_visible(axs.flat, [True, True, True, True], [True, True, True, True])