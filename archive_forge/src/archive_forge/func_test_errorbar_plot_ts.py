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
@pytest.mark.slow
@pytest.mark.parametrize('yerr', [Series(np.abs(np.random.default_rng(2).standard_normal(12))), DataFrame(np.abs(np.random.default_rng(2).standard_normal((12, 2))), columns=['x', 'y'])])
def test_errorbar_plot_ts(self, yerr):
    ix = date_range('1/1/2000', '1/1/2001', freq='ME')
    ts = Series(np.arange(12), index=ix, name='x')
    yerr.index = ix
    ax = _check_plot_works(ts.plot, yerr=yerr)
    _check_has_errorbars(ax, xerr=0, yerr=1)