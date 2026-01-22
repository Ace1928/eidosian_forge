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
@pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds)
def test_kind_kwarg(self, kind):
    pytest.importorskip('scipy')
    s = Series(range(3))
    _, ax = mpl.pyplot.subplots()
    s.plot(kind=kind, ax=ax)
    mpl.pyplot.close()