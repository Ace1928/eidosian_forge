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
@pytest.mark.parametrize('kwargs', [{}, {'layout': (-1, 1)}, {'layout': (1, -1)}])
def test_plot_6951(self, ts, kwargs):
    ax = _check_plot_works(ts.plot, subplots=True, **kwargs)
    _check_axes_shape(ax, axes_num=1, layout=(1, 1))