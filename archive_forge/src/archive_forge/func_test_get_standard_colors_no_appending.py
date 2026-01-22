import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_get_standard_colors_no_appending(self):
    from matplotlib import cm
    from pandas.plotting._matplotlib.style import get_standard_colors
    color_before = cm.gnuplot(range(5))
    color_after = get_standard_colors(1, color=color_before)
    assert len(color_after) == len(color_before)
    df = DataFrame(np.random.default_rng(2).standard_normal((48, 4)), columns=list('ABCD'))
    color_list = cm.gnuplot(np.linspace(0, 1, 16))
    p = df.A.plot.bar(figsize=(16, 7), color=color_list)
    assert p.patches[1].get_facecolor() == p.patches[17].get_facecolor()