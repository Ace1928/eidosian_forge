from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_bar_bottom_left_subplots(self):
    df = DataFrame(np.random.default_rng(2).random((5, 5)))
    axes = df.plot.bar(subplots=True, bottom=-1)
    for ax in axes:
        result = [p.get_y() for p in ax.patches]
        assert result == [-1] * 5
    axes = df.plot.barh(subplots=True, left=np.array([1, 1, 1, 1, 1]))
    for ax in axes:
        result = [p.get_x() for p in ax.patches]
        assert result == [1] * 5