import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_externally_shared_axes(self):
    df = DataFrame({'a': np.random.default_rng(2).standard_normal(1000), 'b': np.random.default_rng(2).standard_normal(1000)})
    fig = mpl.pyplot.figure()
    plots = fig.subplots(2, 3)
    plots[0][0] = fig.add_subplot(231, sharex=plots[1][0])
    plots[0][2] = fig.add_subplot(233, sharex=plots[1][2])
    twin_ax1 = plots[0][1].twinx()
    twin_ax2 = plots[0][2].twinx()
    df['a'].plot(ax=plots[0][0], title='External share only').set_xlabel('this label should never be visible')
    df['a'].plot(ax=plots[1][0])
    df['a'].plot(ax=plots[0][1], title='Internal share (twin) only').set_xlabel('this label should always be visible')
    df['a'].plot(ax=plots[1][1])
    df['a'].plot(ax=plots[0][2], title='Both').set_xlabel('this label should never be visible')
    df['a'].plot(ax=plots[1][2])
    df['b'].plot(ax=twin_ax1, color='green')
    df['b'].plot(ax=twin_ax2, color='yellow')
    assert not plots[0][0].xaxis.get_label().get_visible()
    assert plots[0][1].xaxis.get_label().get_visible()
    assert not plots[0][2].xaxis.get_label().get_visible()