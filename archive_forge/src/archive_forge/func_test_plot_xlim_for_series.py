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
@pytest.mark.parametrize('kind', ['line', 'area'])
def test_plot_xlim_for_series(self, kind):
    s = Series([2, 3])
    _, ax = mpl.pyplot.subplots()
    s.plot(kind=kind, ax=ax)
    xlims = ax.get_xlim()
    assert xlims[0] < 0
    assert xlims[1] > 1