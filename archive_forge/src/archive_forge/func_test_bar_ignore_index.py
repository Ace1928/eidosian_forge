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
def test_bar_ignore_index(self):
    df = Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    _, ax = mpl.pyplot.subplots()
    ax = df.plot.bar(use_index=False, ax=ax)
    _check_text_labels(ax.get_xticklabels(), ['0', '1', '2', '3'])