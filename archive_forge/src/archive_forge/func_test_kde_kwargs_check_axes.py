from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_kde_kwargs_check_axes(self, ts):
    pytest.importorskip('scipy')
    _, ax = mpl.pyplot.subplots()
    sample_points = np.linspace(-100, 100, 20)
    ax = ts.plot.kde(logy=True, bw_method=0.5, ind=sample_points, ax=ax)
    _check_ax_scales(ax, yaxis='log')
    _check_text_labels(ax.yaxis.get_label(), 'Density')