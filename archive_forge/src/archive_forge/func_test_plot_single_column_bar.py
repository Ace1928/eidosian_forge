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
@pytest.mark.slow
@pytest.mark.parametrize('layout', [None, (-1, 1)])
def test_plot_single_column_bar(self, layout):
    df = DataFrame({'x': np.random.default_rng(2).random(10)})
    axes = _check_plot_works(df.plot.bar, subplots=True, layout=layout)
    _check_axes_shape(axes, axes_num=1, layout=(1, 1))