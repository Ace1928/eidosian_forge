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
def test_label_none(self):
    s = Series([1, 2])
    _, ax = mpl.pyplot.subplots()
    ax = s.plot(legend=True, ax=ax)
    _check_legend_labels(ax, labels=[''])
    mpl.pyplot.close('all')