import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_has_externally_shared_axis_y_axis(self):
    func = plotting._matplotlib.tools._has_externally_shared_axis
    fig = mpl.pyplot.figure()
    plots = fig.subplots(4, 2)
    plots[0][0] = fig.add_subplot(321, sharey=plots[0][1])
    plots[2][0] = fig.add_subplot(325, sharey=plots[2][1])
    plots[1][0].twiny()
    plots[2][0].twiny()
    assert func(plots[0][0], 'y')
    assert not func(plots[1][0], 'y')
    assert func(plots[2][0], 'y')
    assert not func(plots[3][0], 'y')