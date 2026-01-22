import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_plot(close_figures):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        hdrboxplot(data, labels=labels.tolist(), ax=ax, threshold=1, seed=12345)
        ax.set_xlabel('Month of the year')
        ax.set_ylabel('Sea surface temperature (C)')
        ax.set_xticks(np.arange(13, step=3) - 1)
        ax.set_xticklabels(['', 'Mar', 'Jun', 'Sep', 'Dec'])
        ax.set_xlim([-0.2, 11.2])
    except OSError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')